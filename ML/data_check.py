import pickle
from collections import defaultdict

with open(f"./data/amazon/Movies_and_TV_text_name_dict.json.gz", "rb") as ft:
    text_name_dict = pickle.load(ft)


def find_missing_numbers(numbers, desc_numbers, start, end):
    # 전체 범위 집합
    full_set = set(range(start, end + 1))

    # 리스트에서 실제로 있는 숫자 집합
    number_set = set(numbers)
    desc_number_set = set(desc_numbers)

    # 빠진 숫자 찾기
    missing_numbers = sorted(full_set - number_set)
    desc_missing_numbers = sorted(full_set - desc_number_set)

    return missing_numbers, desc_missing_numbers


def load_data(fname, max_len, path=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    data = {}

    # assume user/item index starting from 1

    # f = open('./pre_train/sasrec/data/%s.txt' % fname, 'r')
    if path == None:
        f = open("./data/amazon/%s.txt" % fname, "r")
    else:
        f = open(path, "r")
    for line in f:
        u, i = line.rstrip().split(" ")
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)  # id가 곧 유저의 개수
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        seq = [0] * max_len
        length_idx = max_len - 1

        for i in reversed(User[user][:-1]):
            seq[length_idx] = i
            length_idx -= 1
            if length_idx == -1:
                break

        data[user] = seq

    return data


# Missing item을 포함하는 유저와 아이템 찾기
def find_users_with_missing_items(dataset, missing_items):
    users_with_missing_items = defaultdict(list)

    # 각 유저의 시퀀스를 순회
    for user, sequence in dataset.items():
        # 시퀀스와 missing item의 교집합 찾기
        intersecting_items = set(sequence) & set(missing_items)
        if intersecting_items:
            # 교집합이 있다면 해당 유저와 아이템 저장
            users_with_missing_items[user] = intersecting_items

    return users_with_missing_items


missing, desc_missing = find_missing_numbers(
    list(text_name_dict["title"].keys()),
    list(text_name_dict["description"].keys()),
    1,
    86678,
)
dataset = load_data("Movies_and_TV", 50)
users_with_missing = find_users_with_missing_items(dataset, missing)

# 결과 출력
print(len(list(users_with_missing.keys())))
print(missing)
print()
print(users_with_missing[117])
print()
print(dataset[117])
