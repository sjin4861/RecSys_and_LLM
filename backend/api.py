import argparse
import os
import random
import sys
import time
from contextlib import asynccontextmanager

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from backend.inference import ModelManager
from ML.models.ALLMRec.a_llmrec_model import *
from ML.utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MONGO_URI = "mongodb://localhost:27017/"

# model_manager = None
# client = None
# db = None
# user = None
# item = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 시 실행할 코드"""
    global model_manager, client, db, user, item

    # MongoDB 연결
    client = MongoClient(MONGO_URI)
    db = client["items"]
    user = db["user"]
    item = db["item"]

    cold_items = find_cold(user, 50)
    text_name_dict = get_text_name_dict(item)
    missing_list = get_missing(text_name_dict["title"])
    data = [cold_items, text_name_dict, missing_list]

    # 모델 로드
    model_manager = ModelManager(data)

    yield  # 애플리케이션 실행

    # 리소스 정리
    model_manager = None
    client.close()
    print("서버 종료: 모델 리소스 해제 및 MongoDB 연결 종료.")


# FastAPI 앱 생성 (lifespan 적용)
app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    user_id: int


@app.post("/predict-main")
def predict(
    request: PredictRequest, model: ModelManager = Depends(lambda: model_manager)
):
    user_id = request.user_id
    user_data = user.find_one({"_id": user_id})

    if user_data:
        seq = [item["itemnum"] for item in user_data.get("items", [])]
    else:
        return {"error": "존재하지 않는 유저입니다."}

    if model is None:
        return {"error": "모델이 로드되지 않았습니다."}

    # 모델 기반 추론 수행
    result = model.inference(user_id, seq)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
