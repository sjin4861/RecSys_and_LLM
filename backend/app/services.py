from backend.app.schemas import *


def sign_up(request: SignUpRequest, user_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if user_data:
        return {"error": "이미 존재하는 아이디입니다."}

    new_user = {
        "_id": str(user_collection.count_documents({}) + 1),  # UserNum 자동 생성
        "reviewerID": request.reviewer_id,
        "password": request.password,
        "userName": request.name,
        "items": [],
    }

    user_collection.insert_one(new_user)
    return {"message": "회원가입 성공", "user_id": new_user["_id"]}


def sign_in(request: SignInRequest, model_manager, user_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return {"error": "존재하지 않는 사용자입니다."}

    if user_data["password"] != request.password:
        return {"error": "비밀번호가 일치하지 않습니다."}

    user_id = user_data["_id"]
    seq = [item["itemnum"] for item in user_data.get("items", [])]

    if model_manager is None:
        return {"error": "모델이 로드되지 않았습니다."}

    result = model_manager.inference(user_id, seq)
    # parse result

    return {"message": "로그인 성공", "user_id": user["_id"], "name": user["userName"]}


def detail_prediction(
    request: DetailPredictRequest, model_manager, item_collection, review_collection
):
    item_data = item_collection.find_one({"_id": request.item_id})

    if not item_data:
        return {"error": "존재하지 않는 아이템입니다."}

    # 필요한 정보 추출
    review_data = review_collection.find_one({"_id": item_data["_id"]})
    reviews_dict = review_data.get("review", {}) if review_data else {}

    reviews = []
    for reviewer_name, review_text in reviews_dict.items():
        reviews.append({"user_name": reviewer_name, "review": review_text})

    title = item_data.get("title", "No Title Available")
    cast = item_data.get("cast", [])
    description_list = item_data.get("description", [])
    if isinstance(description_list, list) and description_list:
        description = max(description_list, key=len)
    else:
        description = "No Description Available"

    # 모델 추론
    predictions = []

    # 5. 반환할 데이터 구성
    return {
        "title": title,
        "cast": cast,
        "description": description,
        "reviews": reviews,
        "prediction": predictions,
    }


# for test
def main_prediction(request: MainPredictRequest, model_manager, user_collection):
    user_data = user_collection.find_one({"_id": request.user_id})

    if not user_data:
        return {"error": "존재하지 않는 유저입니다."}

    seq = [item["itemnum"] for item in user_data.get("items", [])]

    if model_manager is None:
        return {"error": "모델이 로드되지 않았습니다."}

    result = model_manager.inference(user_id, seq)

    # parse

    return result
