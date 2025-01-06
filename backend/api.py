import argparse
import os
import random
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from fastapi import FastAPI
from model import llmrec_args, setup_ddp
from models.a_llmrec_model import *
from pre_train.sasrec.utils import *
from pydantic import BaseModel
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# FastAPI 앱 생성
app = FastAPI()


# 요청 데이터 구조 정의
class PredictionRequest(BaseModel):
    input: int  # 단일 실수 입력


def inference(rank, world_size, args, user_id):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = "cuda:" + str(rank)

    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = 10
    phase2_epoch = 10
    model.load_model(args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch)

    dataset = data_partition(
        args.rec_pre_trained_data, path=f"./data/amazon/{args.rec_pre_trained_data}.txt"
    )
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print("user num:", usernum, "item num:", itemnum)

    if user_id <= 0 or user_id > usernum:
        print("No such user")
        return

    num_batch = len(user_train) // args.batch_size_infer
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))
    model.eval()

    users = range(1, usernum + 1)

    user_list = []
    for u in users:
        if len(user_train[u]) < 1 or len(user_test[u]) < 1:
            continue
        user_list.append(u)

    inference_data_set = SeqDataset_Inference(
        user_train, user_valid, user_test, user_list, itemnum, args.maxlen
    )

    u, seq, neg = inference_data_set[user_id - 1]
    u = np.expand_dims(u, axis=0)
    seq = np.expand_dims(seq, axis=0)
    neg = np.expand_dims(neg, axis=0)
    # u, seq, neg = u.numpy(), seq.numpy(), neg.numpy()
    result = model([u, seq, neg, rank], mode="generate")

    return result


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
        prediction = inference(0, 0, llmrec_args, user_id)
    return {"input": request.input, "prediction": prediction}
