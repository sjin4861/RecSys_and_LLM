import re
from datetime import datetime

import requests

from backend.app.config import DEFAULT_IMAGE_URL
from backend.app.schemas import *


def get_item_img(url_lst):
    if not url_lst:  # 리스트가 비어있으면 기본 이미지 반환
        return DEFAULT_IMAGE_URL

    for url in url_lst:
        try:
            response = requests.head(url, timeout=3)  # HEAD 요청으로 빠르게 확인
            if response.status_code == 200:
                return url
        except requests.RequestException:
            continue

    return DEFAULT_IMAGE_URL


def sign_up(request: SignUpRequest, user_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if user_data:
        return ApiResponse(success=False, message="이미 존재하는 아이디입니다.")

    new_user = {
        "_id": str(user_collection.count_documents({}) + 1),  # UserNum 자동 생성
        "reviewerID": request.reviewer_id,
        "password": request.password,
        "userName": request.name,
        "items": [],
    }

    user_collection.insert_one(new_user)
    return ApiResponse(
        success=True, message="회원가입 성공", data={"user_id": new_user["_id"]}
    )


def sign_in(request: SignInRequest, model_manager, user_collection, item_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 사용자입니다.")

    if user_data["password"] != request.password:
        return ApiResponse(success=False, message="비밀번호가 일치하지 않습니다.")

    user_id = user_data["_id"]
    seq = [item["itemnum"] for item in user_data.get("items", [])]
    seq_time = [
        (item["itemnum"], item["unixReviewTime"]) for item in user_data.get("items", [])
    ]

    if model_manager is None:
        return ApiResponse(success=False, message="모델이 로드되지 않았습니다.")

    result = model_manager.inference(user_id, seq, seq_time)

    # match = re.search(r"\(ID: (\d+)\)", result["allmrec_prediction"])
    # allmrec_ids = [match.group(1)] if match else []

    allmrec_ids = [str(result["allmrec_prediction"])]
    gsasrec_ids = list(map(str, result["gsasrec_prediction"]))
    tisasrec_ids = list(map(str, result["tisasrec_prediction"]))

    all_ids = list(set(allmrec_ids + gsasrec_ids + tisasrec_ids))  # 중복 제거
    items = item_collection.find(
        {"_id": {"$in": all_ids}}, {"_id": 1, "available_images": 1}
    )

    # 🔹 결과 가공 (MongoDB에서 찾은 데이터를 Dict으로 변환)
    item_map = {
        item["_id"]: get_item_img(item.get("available_images", [])) for item in items
    }

    # 🔹 JSON 형태로 변환 (요청 형식에 맞게 조정)
    predictions = {
        "prediction-1": {
            "item_id": allmrec_ids[0],
            "img_url": item_map.get(allmrec_ids[0], None),
        },
        "prediction-2": [
            {"item_id": _id, "img_url": item_map.get(_id, None)} for _id in gsasrec_ids
        ],
        "prediction-3": [
            {"item_id": _id, "img_url": item_map.get(_id, None)} for _id in tisasrec_ids
        ],
        "prediction-4": [],  # 장르 모델 완성 후 추가 예정
    }

    return ApiResponse(
        success=True,
        message="로그인 성공",
        data={
            "user_id": user_id,
            "name": user_data["userName"],
            "predictions": predictions,
        },
    )


def detail_prediction(
    request: DetailPredictRequest, model_manager, item_collection, review_collection
):
    item_data = item_collection.find_one({"_id": request.item_id})

    if not item_data:
        return ApiResponse(success=False, message="존재하지 않는 아이템입니다.")

    img = get_item_img(item_data["available_images"])

    # 필요한 정보 추출
    review_data = review_collection.find_one({"_id": item_data["_id"]})
    reviews_dict = review_data.get("review", {}) if review_data else {}

    reviews = []
    for reviewer_name, review_text in reversed(reviews_dict.items()):
        reviews.append({"user_name": reviewer_name, "review": review_text})

    title = item_data.get("title", "No Title Available")
    cast = item_data.get("cast", [])
    description_list = item_data.get("description", [])
    if isinstance(description_list, list) and description_list:
        description = max(description_list, key=len)
    else:
        description = "No Description Available"

    # 모델 추론 - 찬미님 모델
    predictions = []

    # 5. 반환할 데이터 구성
    return ApiResponse(
        success=True,
        message="아이템 정보 조회 성공",
        data={
            "img": img,
            "title": title,
            "cast": cast,
            "description": description,
            "reviews": reviews,
            "predictions": predictions,
        },
    )


def review_post(
    request: ReviewPostRequest, user_collection, item_collection, review_collection
):
    # 존재 여부 체크
    item_data = item_collection.find_one({"_id": request.item_id})

    if not item_data:
        return ApiResponse(success=False, message="존재하지 않는 아이템입니다.")

    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 유저입니다.")

    # 리뷰 업데이트
    user_name = user_data["userName"]
    review_data = review_collection.find_one({"_id": item_data["_id"]})
    if not review_data:
        default_review_data = {
            "_id": item_data["_id"],  # ItemNum
            "review": {},
            "summary": {},
        }
        review_collection.insert_one(default_review_data)
    else:
        review_collection.update_one(
            {"_id": item_data["_id"]},
            {
                "$set": {f"review.{user_name}": request.review}
            },  # `review` 딕셔너리에 값 추가
        )

    # 유저 시퀀스 업데이트
    new_item = {
        "itemnum": item_data["_id"],
        "asin": item_data["asin"],
        "reviewText": request.review,
        "overall": request.rating,
        "summary": "",
        "unixReviewTime": int(datetime.utcnow().timestamp()),  # 현재 시간 기준
    }

    user_collection.update_one(
        {"_id": user_data["_id"]}, {"$push": {"items": new_item}}
    )

    return ApiResponse(success=True, message="리뷰 작성 성공")


# for test
def main_prediction(
    request: MainPredictRequest, model_manager, user_collection, item_collection
):
    user_data = user_collection.find_one({"_id": request.user_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 유저입니다.")

    seq = [item["itemnum"] for item in user_data.get("items", [])]
    seq_time = [
        (item["itemnum"], item["unixReviewTime"]) for item in user_data.get("items", [])
    ]
    if model_manager is None:
        return ApiResponse(success=False, message="모델이 로드되지 않았습니다.")

    result = model_manager.inference(user_data["_id"], seq, seq_time)

    # match = re.search(r"\(ID: (\d+)\)", result["allmrec_prediction"])
    # allmrec_ids = [match.group(1)] if match else []

    allmrec_ids = [str(result["allmrec_prediction"])]
    gsasrec_ids = list(map(str, result["gsasrec_prediction"]))
    tisasrec_ids = list(map(str, result["tisasrec_prediction"]))

    all_ids = list(set(allmrec_ids + gsasrec_ids + tisasrec_ids))  # 중복 제거
    items = item_collection.find(
        {"_id": {"$in": all_ids}}, {"_id": 1, "available_images": 1}
    )

    # 🔹 결과 가공 (MongoDB에서 찾은 데이터를 Dict으로 변환)
    item_map = {
        item["_id"]: get_item_img(item.get("available_images", [])) for item in items
    }

    # 🔹 JSON 형태로 변환 (요청 형식에 맞게 조정)
    predictions = {
        "prediction-1": {
            "item_id": allmrec_ids[0],
            "img_url": item_map.get(allmrec_ids[0], None),
        },
        "prediction-2": [
            {"item_id": _id, "img_url": item_map.get(_id, None)} for _id in gsasrec_ids
        ],
        "prediction-3": [
            {"item_id": _id, "img_url": item_map.get(_id, None)} for _id in tisasrec_ids
        ],
        "prediction-4": [],  # 장르 모델 완성 후 추가 예정
    }

    return ApiResponse(
        success=True, message="메인 추천 성공", data={"predictions": predictions}
    )
