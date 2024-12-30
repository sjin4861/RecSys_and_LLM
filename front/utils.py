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


def init_session_state():
    st.session_state.user_id = ""
    st.session_state.rec_results = None
