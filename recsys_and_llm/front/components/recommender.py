import os
from typing import AnyStr, List

import streamlit as st
from front.utils.image_utils import show_img


@st.cache_data
def rec_main(item: dict):
    if not item:
        st.warning("추천 결과가 없습니다.")
        return

    st.header(f"{st.session_state.user_name}님 지금 이 영화 어떠세요?")

    # 이미지와 링크 표시
    html = f"""
    <style>
        .image-container {{
            text-align: center;
            margin-top: 20px;
        }}
        .image-container img {{
            width: 400px;
            height: 600px;
            border-radius: 16px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2), 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .image-container img:hover {{
            transform: scale(1.05);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        }}
        .image-container p {{
            margin-top: 10px;
            font-size: 18px;
            color: #444;
            font-weight: bold;
        }}
    </style>
    <div class="image-container">
        <a href={os.environ.get("FRONT_URL")}/item_page/?user={st.session_state.reviewer_id}&name={st.session_state.user_name}&item={item["item_id"]} target = '_self'>
            <img src="{item['img_url']}">
        </a>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.header("")


def rec_line(head: str, predictions: List[AnyStr]):
    if not predictions or len(predictions) == 0:
        st.warning("추천 결과가 없습니다.")
        return

    st.subheader(head)

    k = len(predictions)
    columns = st.columns([0.4] * k)

    for idx, item in enumerate(predictions):
        # item_info fetches details for each item in the predictions
        with columns[idx]:
            show_img(
                item["item_id"],
                item["img_url"],
            )


def search(searchterm: str):
    # TODO : 백 (검색어 -> 검색 결과)
    titles = ["a", "ab", "abc", "abcd", "bcd", "cde"]
    res = []
    for t in titles:
        if searchterm in t:
            res.append(t)
    print(res)
    return res
