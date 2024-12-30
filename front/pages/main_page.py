import streamlit as st
from pyparsing import empty
from utils import show_item

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
empty1, con0, empty2 = st.columns([0.1, 2.0, 0.1])  # title
empty1, con1, con2, con3, con4, con5, empty2 = st.columns(
    [0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1]
)  # item 1-5
empty1, con6, con7, con8, con9, con10, empty2 = st.columns(
    [0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1]
)  # item 6-10
empty1, con11, empty2 = st.columns([0.1, 2.0, 0.1])  # empty line
empty1, con12, _, con13, empty2 = st.columns(
    [0.1, 0.15, 1.6, 0.25, 0.1]
)  # prev/next button
empty1, con14, empty2 = st.columns([0.725, 0.15, 0.725])  # logout button

# dummy data
temp_data = {
    i: (
        {
            "item_title": "파묘",
            "img_url": "https://image.tmdb.org/t/p/original/ca5UAt0YJAkrcPlcOUftXnEa6C5.jpg",
            "item_info": "https://pedia.watcha.com/ko-KR/contents/m53mNG2",
            "people": ", ".join(["김고은", "최민식", "유해진", "이도현"]),
            "release": "2024.12.26",
            "avg_score": 4.8,
        }
        if i % 2 == 0
        else {
            "item_title": "짱구",
            "img_url": "https://i.namu.wiki/i/lT70-Fgm_Iw2bPlvJmpKy1QRqEeZiXvsq3Ckw2nMs2eYYmuesH68oWP4gFsAYEFIQZ8zuCXc8Yd0j4f0tc7TZg.webp",
            "item_info": "https://pedia.watcha.com/ko-KR/contents/mdj2e4q",
            "people": ", ".join(["짱구", "흰둥이", "맹구", "철수"]),
            "release": "2024.12.27",
            "avg_score": 3.8,
        }
    )
    for i in range(10)
}


# @st.cache_data
def main():
    if st.session_state.user_id == "":
        st.switch_page("./app.py")
    with empty1:
        empty()
    if st.session_state.rec_results is None:
        rec_results = temp_data
        st.session_state.rec_results = rec_results
    else:
        rec_results = st.session_state.rec_results

    with con0:
        st.header(f"{st.session_state.user_id}님을 위한 추천")
    if rec_results is not None:
        with con1:
            show_item(rec_results[0])
        with con2:
            show_item(rec_results[1])
        with con3:
            show_item(rec_results[2])
        with con4:
            show_item(rec_results[3])
        with con5:
            show_item(rec_results[4])
        with con6:
            show_item(rec_results[5])
        with con7:
            show_item(rec_results[6])
        with con8:
            show_item(rec_results[7])
        with con9:
            show_item(rec_results[8])
        with con10:
            show_item(rec_results[9])
        with con12:
            st.session_state.user_id = ""
            st.session_state.rec_results = None
            st.page_link(page="./app.py", label="처음으로", icon="🏠")
        with con14:
            st.header("")
            st.header("")

            if st.button("Logout"):
                st.switch_page("./app.py")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
