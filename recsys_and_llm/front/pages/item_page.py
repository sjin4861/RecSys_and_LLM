import streamlit as st  # type: ignore
from front.components.recommender import rec_line
from front.utils.item_utils import get_detail, show_info
from pyparsing import empty

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
rec_data_8 = [str(0) for _ in range(4)] + [str(1) for _ in range(4)]  # 데모데이터


def get_rec_res(user_id: str):
    # 데모 데이터에서 돌아가게 하기 위한 임시 함수 | 백 연결 후 삭제 #
    rec_data_1 = str(2)
    return {"main": rec_data_1, "line": rec_data_8}


def main():
    st.session_state.user_id = st.query_params["user"]
    st.session_state.selected = st.query_params["item"]
    st.session_state.rec_results = get_rec_res(
        st.session_state.user_id
    )  # TODO : 백 (user_id -> 추천 결과)

    if st.session_state.user_id == "":
        st.switch_page("./app.py")

    empty1, _, empty2 = st.columns([0.1, 2.0, 0.1])
    with empty1:
        empty()

    if st.session_state.selected is not None:
        show_info(st.session_state.selected)
        st.header("")
        rec_line(
            f"{get_detail(st.session_state.selected)['item_title']}와 유사한 작품",
            rec_data_8,
        )

        _, back_btn, logout_btn, _ = st.columns([0.725, 0.15, 0.15, 0.725])

        with back_btn:
            st.header("")

            if st.button("Back"):
                st.switch_page("./pages/main_page.py")

        with logout_btn:
            st.header("")

            if st.button("Logout"):
                st.switch_page("./app.py")

    with empty2:
        empty()


if __name__ == "__main__":
    main()
