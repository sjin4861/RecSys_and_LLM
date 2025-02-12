import os
from datetime import datetime

import numpy as np
from pytz import timezone


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)

    return file_paths


def get_missing(title_id_dict):
    exist_ids = np.array(list(title_id_dict.keys()))

    return np.setdiff1d(
        np.arange(1, max(exist_ids) + 1, dtype=np.int32), exist_ids, assume_unique=True
    ).tolist()


def find_cold(user_collection, max_len):
    data = {}

    # db data -> dict
    for user_data in user_collection.find():
        user_id = str(user_data["_id"])  # _id를 문자열로 변환
        itemnums = [
            item["itemnum"] for item in user_data.get("items", [])
        ]  # itemnum 리스트 추출

        # 최신 max_len 개수만 유지 (패딩 포함)
        seq = [0] * max_len
        recent_items = itemnums[
            -max_len:
        ]  # 마지막 아이템을 제외한 최신 max_len개만 유지
        seq[-len(recent_items) :] = recent_items  # 패딩 포함하여 우측 정렬

        data[user_id] = seq  # 결과 저장

    item_interactions = []
    for user, items in data.items():
        item_interactions.extend(items)

    # Step 2: Count occurrences of each item
    item_counts = Counter(item_interactions)

    # Step 3: Sort items by interaction count
    sorted_items = sorted(
        item_counts.items(), key=lambda x: x[1], reverse=True
    )  # (item, count)

    # Step 4: Calculate thresholds for warm and cold items
    total_items = len(sorted_items)
    cold_threshold = int(total_items * 0.35)  # Bottom 35%

    # Get cold items (bottom 35% of interactions)
    cold_items = [item for item, count in sorted_items[-cold_threshold:]]

    return cold_items


def get_text_name_dict(item_collection):
    text_name_dict = {"title": {}, "description": {}}

    for item in item_collection.find():
        item_id = str(item["_id"])  # _id를 문자열로 변환
        title = item.get("title", "No Title")  # 기본값 설정
        description = item.get("description", ["No Description"])  # 기본값 설정

        # 리스트 타입이면 문자열로 변환
        if isinstance(description, list):
            description = " ".join(description) if description else "No Description"

        text_name_dict["title"][item_id] = title
        text_name_dict["description"][item_id] = description

    return text_name_dict


def make_candidate_for_LLM(model, itemnum, log_seq, args):
    seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    for i in reversed(log_seq):
        seq[idx] = i
        idx -= 1
        if idx == -1:
            break
    rated = set(seq)
    rated.add(0)

    item_idx = []
    for t in range(1, itemnum + 1):
        if t in rated:
            continue
        item_idx.append(t)

    predictions = -model.predict(*[np.array(l) for l in [[-1], [seq], item_idx]])
    predictions = predictions[0]  # - for 1st argsort DESC

    # Top-K 아이템 선택 (가장 높은 점수 기준)
    top_k = 10
    top_k_indices = predictions.argsort()[
        :top_k
    ]  # 점수가 높은 Top-K 아이템의 인덱스 선택

    # 실제 아이템 번호로 변환
    top_k_items = [
        item_idx[idx] for idx in top_k_indices
    ]  # 인덱스를 아이템 번호로 매핑

    return top_k_items
