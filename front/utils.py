import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from st_clickable_images import clickable_images

item_info_map = {
    0: {
        "item_title": "파묘",
        "img_url": "https://image.tmdb.org/t/p/original/ca5UAt0YJAkrcPlcOUftXnEa6C5.jpg",
        "item_info": "https://pedia.watcha.com/ko-KR/contents/m53mNG2",
        "people": ", ".join(["김고은", "최민식", "유해진", "이도현"]),
        "release": "2024.12.26",
        "avg_score": 4.8,
    },
    1: {
        "item_title": "짱구",
        "img_url": "https://i.namu.wiki/i/lT70-Fgm_Iw2bPlvJmpKy1QRqEeZiXvsq3Ckw2nMs2eYYmuesH68oWP4gFsAYEFIQZ8zuCXc8Yd0j4f0tc7TZg.webp",
        "item_info": "https://pedia.watcha.com/ko-KR/contents/mdj2e4q",
        "people": ", ".join(["짱구", "흰둥이", "맹구", "철수"]),
        "release": "2024.12.27",
        "avg_score": 3.0,
    },
    2: {
        "item_title": "친절한 금자씨",
        "img_url": "https://web-cf-image.cjenm.com/crop/660x950/public/share/metamng/programs/sympathyforladyvengeance-movie-poster-ko-001-29.jpg?v=1677040153",
        "item_info": "https://watcha.com/contents/mdRL9gW?search_id=8cc236f3-61e8-43bd-9288-3415cfbc6591",
        "people": ", ".join(["이영애", "최민식", "라미란", "김병옥"]),
        "release": "2025.01.13",
        "avg_score": 4.9,
    },
}


@st.cache_resource
def show_img(
    item_title: str = "title",
    img_url: str = "https://ghcci.korcham.net/images/no-image01.gif",
    info_url: str = None,
):
    image = Image.open(BytesIO(requests.get(img_url, stream=True).content)).resize(
        (600, 900)
    )
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    html = (
        f"<a href='{info_url}'><img srcimg src='data:image/png;base64,{img_str}'></a>"
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource
def show_info(item_info):
    show_img(item_info["item_title"], item_info["img_url"], item_info["item_info"])
    _, temp, _ = st.columns([0.3, 0.5, 0.3])
    with temp:
        st.markdown(f"출연진 : {item_info['people']}")
        st.markdown(f"개봉일 : {item_info['release']}")
        st.markdown(f"평균 평점 : {item_info['avg_score']}")
        st.markdown(f"[더보기]({item_info['item_info']})")


def get_detail(item_id: int = 1):
    return item_info_map[item_id]


def init_session_state():
    st.session_state.user_id = ""
    st.session_state.rec_results = None
    st.session_state.selected = None
