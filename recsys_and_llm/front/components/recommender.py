import os
from typing import AnyStr, List

import streamlit as st
from front.utils.image_utils import show_img
from front.utils.item_utils import get_detail


@st.cache_data
def rec_main(user_id: str, item_id: str):
    if not item_id:
        st.warning("추천 결과가 없습니다.")
        return

    item_info = get_detail(item_id)
    st.header(f"{user_id}님 지금 이 영화 어떠세요?")

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
        <a href={os.environ.get("BASE_URL")}/item_page/?user={st.session_state.user_id}&item={item_id} target = '_self'>
            <img src="{item_info['img_url']}">
        </a>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.header("")


def rec_line(head: str, rec_results: List[AnyStr]):
    if not rec_results or len(rec_results) == 0:
        st.warning("추천 결과가 없습니다.")
        return

    st.subheader(head)

    k = len(rec_results)
    columns = st.columns([0.4] * k)

    for idx, item_id in enumerate(rec_results):
        # item_info fetches details for each item in the rec_results
        with columns[idx]:
            item_info = get_detail(item_id)
            show_img(
                item_id,
                item_info["img_url"],
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


@st.dialog("More information")
def info_modal(item_id: str = None):
    item_info = get_detail(item_id)
    st.image(item_info["img_url"])
    st.header(item_info["item_title"])
    st.text(f"출연진: {item_info['people']}")
    st.text(f"개봉일: {item_info['release']}")
    st.text(f"평균 평점: {item_info['avg_score']}")
