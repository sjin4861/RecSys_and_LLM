from io import BytesIO

import requests
import streamlit as st
from PIL import Image


@st.cache_resource
def show_img(
    item_title: str = "title",
    img_url: str = "https://ghcci.korcham.net/images/no-image01.gif",
):
    image = Image.open(BytesIO(requests.get(img_url, stream=True).content)).resize(
        (600, 900)
    )
    st.image(
        image,
        caption=item_title,
        use_container_width=True,
    )


@st.cache_resource
def show_item(item_info):
    show_img(item_info["item_title"], item_info["img_url"])
    _, temp, _ = st.columns([0.37, 0.4, 0.23])
    with temp:
        with st.popover("info"):
            st.markdown(f"출연진 : {item_info['people']}")
            st.markdown(f"개봉일 : {item_info['release']}")
            st.markdown(f"평균 평점 : {item_info['avg_score']}")
            st.markdown(f"[더보기]({item_info['item_info']})")


def init_session_state():
    st.session_state.user_id = ""
    st.session_state.rec_results = None
