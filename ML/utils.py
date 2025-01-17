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
