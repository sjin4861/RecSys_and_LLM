import argparse
import os
import random
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from fastapi import FastAPI
from inference import inference, llmrec_args
from pydantic import BaseModel
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ML.models.a_llmrec_model import *
from ML.pre_train.sasrec.utils import *

# FastAPI 앱 생성
app = FastAPI()


# 요청 데이터 구조 정의
class PredictionRequest(BaseModel):
    input: int  # 단일 실수 입력


model = A_llmrec_model(llmrec_args).to(llmrec_args.device)
phase1_epoch = 10
phase2_epoch = 10
model.load_model(llmrec_args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch)
model.eval()


# 기본 엔드포인트 정의
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend!"}


# 엔드포인트 정의
@app.post("/predict")
def predict(request: PredictionRequest):
    # 입력 데이터를 PyTorch 텐서로 변환
    user_id = torch.tensor([[request.input]], dtype=torch.int32)
    # 모델 예측
    with torch.no_grad():
        prediction = inference(user_id, model)
    return {"input": request.input, "prediction": prediction}
