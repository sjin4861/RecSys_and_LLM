import argparse
import os
import pickle
import random
import time
from argparse import Namespace
from logging import getLogger

import numpy as np
import torch
import torch.multiprocessing as mp
from huggingface_hub import hf_hub_download, snapshot_download
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ML.models.ALLMRec.a_llmrec_model import *
from ML.models.ALLMRec.pre_train.sasrec.utils import *
from ML.models.LGCN.model import MLGCN


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def load_data_and_model(model_file, data_path):
    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    config["data_path"] = data_path
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    config["model"] = "MLGCN" if config["model"] == "LightGCN" else "MODEL NAME ERROR"
    print(f"######### LOAD MODEL : {config["model"]} #########")
    if config["model"] == "MLGCN":
        model = MLGCN
    else:
        raise ValueError("`model_name` [{}] is not the name of an existing model.")
    init_seed(config["seed"], config["reproducibility"])
    model = model(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data


def prepare(model_version: str = "LightGCN-Jan-08-2025_10-28-58"):
    data_path = snapshot_download(
        repo_id="PNUDI/recbole",
        cache_dir="../ML/data",
        repo_type="dataset",
    )
    model_path = hf_hub_download(
        repo_id="PNUDI/LightGCN",
        filename=f"{model_version}.pth",
        cache_dir="./models/saved_models",
        repo_type="model",
    )
    with open(f"{data_path}/Movies_and_TV_text_name_dict.json.gz", "rb") as f:
        expected_dict = pickle.load(f)

    # model, dataset 불러오기
    _, model, dataset, _, _, _ = load_data_and_model(model_path, data_path)

    return expected_dict["title"], model, dataset


def predict(user_token: str, token2title: dict, model, dataset, topk: int = 10):
    matrix = dataset.inter_matrix(form="csr")
    model.eval()
    user_id = dataset.token2id("user_id", user_token)
    score = model.full_sort_predict(user_id)
    rating_pred = score.cpu().data.numpy().copy()

    interacted_indices = matrix[user_id].indices
    rating_pred[interacted_indices] = 0
    ind = np.argpartition(rating_pred, -topk)[-topk:]

    arr_ind = rating_pred[ind]
    arr_ind_argsort = np.argsort(arr_ind)[::-1]
    batch_pred_list = ind[arr_ind_argsort]
    batch_pred_list = batch_pred_list.astype(str)

    # print(f'user_token : {user_token}, user_id : {user_id}')
    # print(f'item_pred_ids : {batch_pred_list}, item_pred_tokens : {[dataset.token2id('item_id', item) for item in batch_pred_list]}')

    return [token2title[int(ele.item())] for ele in batch_pred_list]


class ModelManager:
    def __init__(self):
        """모델 및 데이터 로드"""
        self.allmrec_model = None
        self.lgcn_model = None
        self.expected_dict = None
        self.lgcn_dataset = None
        self.llmrec_args = Namespace(
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
        self._load_models()

    def _load_models(self):
        """모델 및 데이터 로드 로직"""
        # Load ALLMRec Model

        self.allmrec_model = A_llmrec_model(self.llmrec_args).to(
            self.llmrec_args.device
        )
        phase1_epoch = 10
        phase2_epoch = 10
        self.allmrec_model.load_model(
            self.llmrec_args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch
        )
        self.allmrec_model.eval()

        # Load LGCN Model and Dataset
        self.expected_dict, self.lgcn_model, self.lgcn_dataset = prepare()

    def inference(self, user_id):
        """ALLMRec 및 LGCN 모델 기반 추론"""
        # LGCN 모델 기반 예측
        lgcn_predictions = "no"
        lgcn_predictions = predict(
            str(user_id), self.expected_dict, self.lgcn_model, self.lgcn_dataset
        )
        lgcn_predictions = "\n".join(lgcn_predictions)

        # ALLMRec 모델 기반 예측
        dataset = load_data(
            self.allmrec_model.args.rec_pre_trained_data,
            self.llmrec_args.maxlen,
            path=f"./ML/data/amazon/{self.allmrec_model.args.rec_pre_trained_data}.txt",
        )
        [data, usernum, itemnum] = dataset

        if user_id <= 0 or user_id > usernum:
            return {"error": "Invalid user_id"}

        seq = np.expand_dims(np.array(data[user_id]), axis=0)
        allmrec_prediction = self.allmrec_model([user_id, seq], mode="inference")

        return {
            "lgcn_predictions": lgcn_predictions,
            "allmrec_prediction": allmrec_prediction,
        }
