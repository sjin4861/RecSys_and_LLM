import contextlib
import glob
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

from ML.models.ALLMRec.pre_train.sasrec.model import SASRec
from ML.utils import *


def load_checkpoint(repo_id, recsys, pre_trained):
    api = HfApi()

    # 저장소 내 파일 리스트 가져오기
    file_list = api.list_repo_files(repo_id)

    # 특정 폴더 내 .pth 파일 필터링
    pth_files = [
        f
        for f in file_list
        if f.startswith(f"recsys/{pre_trained}/{recsys}/") and f.endswith(".pth")
    ]

    assert (
        len(pth_files) == 1
    ), "There should be exactly one model file in the directory."

    # 필요한 파일 다운로드 (메모리 버퍼 사용)
    file_name = pth_files[0]
    pth_file_path = hf_hub_download(repo_id=repo_id, filename=file_name)
    kwargs, checkpoint = torch.load(pth_file_path, map_location="cpu")
    logging.info("Loaded checkpoint from Hugging Face: %s" % pth_file_path)
    return kwargs, checkpoint


class RecSys(nn.Module):
    def __init__(self, repo_id, recsys_model, pre_trained_data, device):
        super().__init__()
        kwargs, checkpoint = load_checkpoint(repo_id, recsys_model, pre_trained_data)
        kwargs["args"].device = device
        model = SASRec(**kwargs)
        model.load_state_dict(checkpoint)

        for p in model.parameters():
            p.requires_grad = False

        self.item_num = model.item_num
        self.user_num = model.user_num
        self.model = model.to(device)
        self.hidden_units = kwargs["args"].hidden_units

    def forward():
        print("forward")
