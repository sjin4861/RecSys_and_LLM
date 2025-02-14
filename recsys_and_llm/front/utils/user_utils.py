import time

import streamlit as st


@st.dialog("Sign up")
def signup():
    user_id = st.text_input("Username", label_visibility="visible", key="signup_id")
    password = st.text_input(
        "Password", type="password", label_visibility="visible", key="signup_pw"
    )
    if st.button("Submit"):
        result = True  # TODO : 백 (데이터 -> 회원가입 결과)
        print(user_id, password)
        if result:
            st.text("Your registration is complete!")
            time.sleep(2)
            st.session_state.user_id = user_id
            st.switch_page(
                "./pages/main_page.py",
            )
        else:
            st.text("error msg")  # TODO : 결과 상황에 따른 메세지 출력
