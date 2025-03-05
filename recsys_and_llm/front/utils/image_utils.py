# utils/image_utils.py
import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


@st.cache_data
def show_img(
    item_id: str,
    img_url: str,
    clickable: bool = True,
):
    """
    이미지 URL을 통해 이미지를 로드하고 Streamlit에 표시
    """
    image = Image.open(BytesIO(requests.get(img_url, stream=True).content)).resize(
        (600, 900)
    )
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    html = f"<img src='data:image/png;base64,{img_str}'>"
    html = (
        f"<a href='{os.environ.get('FRONT_URL')}/item_page/?user={st.session_state.reviewer_id}&name={st.session_state.user_name}&item={item_id}' target = '_self'>{html}</a>"
        if clickable
        else html
    )
    st.markdown(html, unsafe_allow_html=True)
