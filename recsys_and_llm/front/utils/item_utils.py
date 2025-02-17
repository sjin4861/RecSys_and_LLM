# utils/item_utils.py
from typing import AnyStr, Dict

import streamlit as st

from .image_utils import show_img

# 아이템 정보 맵 (데이터)
item_info_map = {
    "0": {
        "item_title": "파묘",
        "img_url": "https://image.tmdb.org/t/p/original/ca5UAt0YJAkrcPlcOUftXnEa6C5.jpg",
        "people": ", ".join(["김고은", "최민식", "유해진", "이도현"]),
        "desc": "무덤을 파다",
        "avg_score": 4.8,
    },
    "1": {
        "item_title": "짱구",
        "img_url": "https://i.namu.wiki/i/lT70-Fgm_Iw2bPlvJmpKy1QRqEeZiXvsq3Ckw2nMs2eYYmuesH68oWP4gFsAYEFIQZ8zuCXc8Yd0j4f0tc7TZg.webp",
        "people": ", ".join(["짱구", "흰둥이", "맹구", "철수"]),
        "desc": "떡잎방범대",
        "avg_score": 3.0,
    },
    "2": {
        "item_title": "친절한 금자씨",
        "img_url": "https://web-cf-image.cjenm.com/crop/660x950/public/share/metamng/programs/sympathyforladyvengeance-movie-poster-ko-001-29.jpg?v=1677040153",
        "people": ", ".join(["이영애", "최민식", "라미란", "김병옥"]),
        "desc": "출소한 금자 이야기",
        "avg_score": 4.9,
    },
}

reviews = {
    "이동진": "흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마.",
    "삼동진": "복수는 구원이 될 수 없다 또 다른 죄의식이 잠식해올 뿐.",
    "사동진": "복수는 또 다른 복수를 낳을뿐 달라지는 건 없다지만 그들에게 용서란 너무 잔인한 말이 아닐까?",
    "오동진": "흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마.",
    "육동진": "흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마.",
    "칠동진": "복수는 구원이 될 수 없다 또 다른 죄의식이 잠식해올 뿐.",
    "팔동진": "복수는 또 다른 복수를 낳을뿐 달라지는 건 없다지만 그들에게 용서란 너무 잔인한 말이 아닐까?",
    "구동진": "흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마.",
}  # TODO : 백에서 가져온다고 가정


def get_review():
    _, _, input_con, btn_con = st.columns([0.27, 0.03, 0.6, 0.1])

    with input_con:
        review_input = st.text_input(
            "review",
            label_visibility="collapsed",
            key="review_input",
            placeholder="How was it?",
        )

    with btn_con:
        if st.button("apply") or review_input:
            print(review_input)
            reviews[st.session_state.user_id] = review_input


def show_reviews(reviews: dict[AnyStr, AnyStr]):
    for u, r in reviews.items():
        st.markdown(f":blue-background[**{u}**]", unsafe_allow_html=True)
        st.markdown(f"{r}", unsafe_allow_html=True)


def show_info(item_id: str):
    """
    아이템 정보를 화면에 표시
    """
    item_info = get_detail(item_id)
    img_con, _, info_con, _ = st.columns([0.27, 0.03, 0.6, 0.1])
    with img_con:
        st.text("")
        show_img(item_id, item_info["img_url"], clickable=False)
    with info_con:
        st.header(item_info["item_title"])
        st.markdown(f"**CAST** : {item_info['people']}")
        st.markdown(f"**Description** : {item_info['desc']}")
        st.subheader("Comment")
        # reviews = {
        #     "이동진": "흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마.",
        #     "삼동진": "복수는 구원이 될 수 없다 또 다른 죄의식이 잠식해올 뿐.",
        #     "사동진": "복수는 또 다른 복수를 낳을뿐 달라지는 건 없다지만 그들에게 용서란 너무 잔인한 말이 아닐까?",
        #     "오동진": "흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마.",
        #     "육동진": "흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마.",
        #     "칠동진": "복수는 구원이 될 수 없다 또 다른 죄의식이 잠식해올 뿐.",
        #     "팔동진": "복수는 또 다른 복수를 낳을뿐 달라지는 건 없다지만 그들에게 용서란 너무 잔인한 말이 아닐까?",
        #     "구동진": "흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마. 흡사 윤리학적 실험실 같은 강렬한 설정에 담긴 딜레마.",
        # }  # TODO : 백에서 가져온다고 가정

        with st.container(height=370, border=False):
            show_reviews(reviews)

    get_review()


def get_detail(item_id: str):
    """
    ID를 통해 특정 아이템 정보를 반환
    """
    return item_info_map[item_id]
