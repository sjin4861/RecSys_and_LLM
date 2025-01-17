import os
import pickle
import sys
from logging import getLogger

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed

abs_path = "/".join(os.path.abspath(__file__).split("/")[:4])
sys.path.insert(0, abs_path)

from ML import MLGCN
from ML.utils import get_missing


def load_data_and_model(model_file, data_path):
    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    config["data_path"] = data_path

    dataset = create_dataset(config)
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
        cache_dir="../ML/models/saved_models",
        repo_type="model",
    )
    with open(f"{data_path}/Movies_and_TV_text_name_dict.json.gz", "rb") as f:
        expected_dict = pickle.load(f)

    # model, dataset 불러오기
    _, model, dataset, _, _, _ = load_data_and_model(model_path, data_path)

    return expected_dict["title"], model, dataset


def predict(user_token: str, token2title: dict, model, dataset, topk: int = 20):
    missing_list = get_missing()
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

    # print(f'user_token : {user_token}, user_id : {user_id}')
    # print(f'item_pred_ids : {batch_pred_list}, item_pred_tokens : {[dataset.token2id('item_id', item) for item in batch_pred_list]}')

    return [
        token2title[int(dataset.id2token("item_id", ele).item())]
        for ele in batch_pred_list
        if int(dataset.id2token("item_id", ele).item()) not in missing_list
    ][: topk // 2]


if __name__ == "__main__":
    ### 실행 예시 ###
    expected_dict, model, dataset = prepare()  # 모델 로딩 -> 초기에 한 번만 실행되면 됨
    user_token = "987"  # 추론할 유저 토큰 [1~311143]
    print(
        "\n".join(predict(user_token, expected_dict, model, dataset))
    )  # 추론 및 결과 출력
