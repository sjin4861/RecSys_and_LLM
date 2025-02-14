from contextlib import asynccontextmanager

from fastapi import FastAPI
from pymongo import MongoClient

from backend.config import DB_NAME, MONGO_URI
from backend.inference import ModelManager
from ML.utils import find_cold, get_missing, get_text_name_dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 시 실행할 코드"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    user_collection = db["user"]
    item_collection = db["item"]
    review_collection = db["review"]
    conversation_collection = db["conversation"]

    # 모델 사용 데이터 파싱
    cold_items = find_cold(user_collection, 50)
    text_name_dict = get_text_name_dict(item_collection)
    missing_list = get_missing(text_name_dict["title"])

    data = [cold_items, text_name_dict, missing_list]

    # 모델 로드
    model_manager = ModelManager(data)

    yield {
        "model_manager": model_manager,
        "user": user_collection,
        "item": item_collection,
        "review": review_collection,
        "conversation": conversation_collection,
    }

    # 리소스 정리
    client.close()
    print("서버 종료: 모델 리소스 해제 및 MongoDB 연결 종료.")
