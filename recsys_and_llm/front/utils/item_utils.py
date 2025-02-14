# utils/item_utils.py
import streamlit as st

from .image_utils import show_img

# 아이템 정보 맵 (데이터)
item_info_map = {
    "0": {
        "item_title": "파묘",
        "img_url": "https://image.tmdb.org/t/p/original/ca5UAt0YJAkrcPlcOUftXnEa6C5.jpg",
        "people": ", ".join(["김고은", "최민식", "유해진", "이도현"]),
        "release": "2024.12.26",
        "avg_score": 4.8,
    },
    "1": {
        "item_title": "짱구",
        "img_url": "https://i.namu.wiki/i/lT70-Fgm_Iw2bPlvJmpKy1QRqEeZiXvsq3Ckw2nMs2eYYmuesH68oWP4gFsAYEFIQZ8zuCXc8Yd0j4f0tc7TZg.webp",
        "people": ", ".join(["짱구", "흰둥이", "맹구", "철수"]),
        "release": "2024.12.27",
        "avg_score": 3.0,
    },
    "2": {
        "item_title": "친절한 금자씨",
        "img_url": "https://web-cf-image.cjenm.com/crop/660x950/public/share/metamng/programs/sympathyforladyvengeance-movie-poster-ko-001-29.jpg?v=1677040153",
        "people": ", ".join(["이영애", "최민식", "라미란", "김병옥"]),
        "release": "2025.01.13",
        "avg_score": 4.9,
    },
}


def show_info(item_id: str):
    """
    아이템 정보를 화면에 표시
    """
    item_info = get_detail(item_id)
    _, con, _ = st.columns([0.3, 0.5, 0.3])
    with con:
        show_img(item_id, item_info["img_url"], clickable=False)
        st.subheader(item_info["item_title"])
        st.markdown(f"출연진 : {item_info['people']}")
        st.markdown(f"개봉일 : {item_info['release']}")
        st.markdown(f"평균 평점 : {item_info['avg_score']}")


def get_detail(item_id: str):
    """
    ID를 통해 특정 아이템 정보를 반환
    """
    return item_info_map[item_id]
