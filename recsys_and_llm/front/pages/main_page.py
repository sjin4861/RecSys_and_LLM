import streamlit as st  # type: ignore
from front.components.recommender import rec_line, rec_main, search
from pyparsing import empty
from streamlit_searchbox import st_searchbox

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

# dummy data
rec_data_1 = str(2)
rec_data_8 = [str(0) for _ in range(4)] + [str(1) for _ in range(4)]


def get_rec_res(user_id: str):
    # 데모 데이터에서 돌아가게 하기 위한 임시 함수 | 백 연결 후 삭제 #
    return {"main": rec_data_1, "line": rec_data_8}


# @st.cache_data
def main():
    if "user" in st.query_params.keys():
        st.session_state.user_id = st.query_params["user"]
        st.session_state.rec_results = get_rec_res(st.session_state.user_id)
    else:
        st.query_params["user"] = st.session_state.user_id

    if st.session_state.user_id == "":
        st.switch_page("./app.py")

    empty1, _, empty2 = st.columns([0.1, 2.0, 0.1])  # empty line
    with empty1:
        empty()

    if st.session_state.rec_results is None:
        rec_results = {"main": rec_data_1, "line": rec_data_8}
        st.session_state.rec_results = rec_results
    else:
        rec_results = st.session_state.rec_results

    if rec_results is not None:
        st_searchbox(
            search,
            placeholder="Search ... ",
            key="my_key",
        )

        rec_main(st.session_state.user_id, rec_results["main"])
        rec_line(f"{st.session_state.user_id}님을 위한 추천", rec_results["line"])
        rec_line(f"최근 본 작품과 유사한 작품", rec_results["line"])
        rec_line(
            f"{st.session_state.user_id}님이 좋아하는 공포영화", rec_results["line"]
        )

        _, logout_btn, _ = st.columns([0.725, 0.15, 0.725])

        with logout_btn:
            st.header("")

            if st.button("Logout"):
                st.switch_page("./app.py")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
