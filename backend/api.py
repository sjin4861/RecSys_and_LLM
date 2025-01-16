import argparse
import os
import random
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from backend.inference import ModelManager
from ML.models.ALLMRec.a_llmrec_model import *
from ML.models.ALLMRec.pre_train.sasrec.utils import *

# FastAPI 앱 생성
app = FastAPI()

model_manager = None


class PredictRequest(BaseModel):
    user_id: int


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global model_manager
    model_manager = ModelManager()
    print("모델이 성공적으로 로드되었습니다.")


@app.post("/predict")
def predict(
    request: PredictRequest, model: ModelManager = Depends(lambda: model_manager)
):
    user_id = request.user_id

    if model is None:
        return {"error": "모델이 로드되지 않았습니다."}

    # 모델 기반 추론 수행
    result = model.inference(user_id)
    return result


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 리소스 정리"""
    global model_manager
    model_manager = None
    print("서버가 종료되었습니다. 모델 리소스가 해제되었습니다.")
