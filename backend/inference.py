import argparse
import os
import random
import time
from argparse import Namespace

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ML.models.ALLMRec.a_llmrec_model import *
from ML.models.ALLMRec.pre_train.sasrec.utils import *

# 고정된 args 값 설정
llmrec_args = Namespace(
    multi_gpu=False,  # Multi-GPU 사용 여부
    gpu_num=0,  # GPU 번호
    llm="opt",  # LLM 모델 선택
    recsys="sasrec",  # RecSys 모델 선택
    rec_pre_trained_data="Movies_and_TV",  # 데이터셋 설정
    pretrain_stage1=False,  # Pretrain 단계 1 활성화
    pretrain_stage2=False,  # Pretrain 단계 2 비활성화
    inference=True,  # Inference 비활성화
    batch_size1=32,  # 단계 1 배치 크기
    batch_size2=2,  # 단계 2 배치 크기
    batch_size_infer=2,  # 추론 배치 크기
    maxlen=50,  # 최대 시퀀스 길이
    num_epochs=10,  # 에포크 수
    stage1_lr=0.0001,  # 단계 1 학습률
    stage2_lr=0.0001,  # 단계 2 학습률
    device="cuda:0",  # 디바이스 설정
)


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def inference(user_id, model):
    dataset = data_partition(
        llmrec_args.rec_pre_trained_data,
        path=f"./ML/data/amazon/{llmrec_args.rec_pre_trained_data}.txt",
    )
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print("user num:", usernum, "item num:", itemnum)

    if user_id <= 0 or user_id > usernum:
        print("No such user")
        return

    num_batch = len(user_train) // llmrec_args.batch_size_infer
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    users = range(1, usernum + 1)

    user_list = []
    for u in users:
        if len(user_train[u]) < 1 or len(user_test[u]) < 1:
            continue
        user_list.append(u)

    inference_data_set = SeqDataset_Inference(
        user_train, user_valid, user_test, user_list, itemnum, llmrec_args.maxlen
    )

    u, seq, neg = inference_data_set[user_id - 1]
    u = np.expand_dims(u, axis=0)
    seq = np.expand_dims(seq, axis=0)
    neg = np.expand_dims(neg, axis=0)
    # u, seq, neg = u.numpy(), seq.numpy(), neg.numpy()
    result = model([u, seq, neg], mode="inference")

    return result
