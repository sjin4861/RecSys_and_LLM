# pages/chat_page.py

import streamlit as st  # type: ignore
from front.components.conversation_manager import (
    load_conversation,
    persist_conversation,
    save_conversation,
)
from front.components.pipeline_manager import set_pipeline
from front.components.response_generator import get_unicrs_response
from front.utils.session_utils import (
    check_login,
    initialize_conversations,
    initialize_feedback_submitted,
    initialize_pipeline,
    initialize_saved_conversations,
)
from front.utils.style_utils import load_styles

########################################
# (예시) set_pipeline, 더미 모듈/객체 설정
########################################
# unicrs_rec = "UniCRS_REC"  # Dummy
# unicrs_gen = "UniCRS_GEN"  # Dummy
# gpt_gen = "GPT_GEN"  # Dummy


########################
# 2) 메인 UI (페이지)  #
########################
def main():
    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

    # 0. 로그인 여부 확인
    if not check_login():
        return
    # 세션 초기화
    initialize_conversations()
    initialize_saved_conversations()
    initialize_pipeline(set_pipeline("", "UniCRS_REC", "UniCRS_GEN"))
    initialize_feedback_submitted()

    if "dialog" not in st.session_state:
        st.session_state.dialog = []

    # -------------------------------------------------------
    # (A) 상단 - "모델 선택"을 가장 먼저(최상단) 배치
    # -------------------------------------------------------
    st.markdown("## UniCRS 대화형 추천 테스트")  # 혹은 st.header("모델 선택")
    model_options = {
        "UniCRS (기본)": "default",
        "FillBlank": "blank",
        "Expansion": "expansion",
        "GPT": "gpt",
    }
    selected_model_label = st.selectbox(
        label="", options=list(model_options.keys()), index=0
    )
    chosen_flag = model_options[selected_model_label]
    # st.session_state.pipeline = set_pipeline(chosen_flag, "UniCRS_REC", "UniCRS_GEN")
    st.session_state.pipeline = set_pipeline(chosen_flag)

    # -------------------------------------------------------
    # 상단 버튼들 (오른쪽 정렬)
    # -------------------------------------------------------
    # CSS 스타일 로드
    load_styles()

    # 상단 버튼들
    # st.markdown('<div class="top-right-buttons">', unsafe_allow_html=True)

    # (A2) "저장된 대화 세션 불러오기"
    st.markdown("#### 저장된 대화 불러오기")
    saved_titles = list(st.session_state.saved_conversations.keys())
    if not saved_titles:
        st.write("저장된 대화 세션이 없습니다.")
    else:
        chosen_session = st.selectbox(
            "불러올 세션", saved_titles, key="load_session_select"
        )
        if st.button("불러오기", key="load_convo_button"):
            st.info(load_conversation(chosen_session))
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # (B) "대화 저장하기" (Form) - 대화가 하나 이상일 때
    # -------------------------------------------------------
    if len(st.session_state.conversations) > 0:
        with st.form("save_form", clear_on_submit=True):
            st.write("### 대화 저장하기")
            new_title = st.text_input("대화 세션 제목을 입력하세요.")
            submitted = st.form_submit_button("저장하기")
            if submitted:
                st.info(save_conversation(new_title))
                persist_conversation(
                    st.session_state.user_id
                )  # <<< 저장시점에 파일로도 기록

    st.write("---")

    # -------------------------------------------------------
    # (C) 채팅 진행
    # -------------------------------------------------------
    user_message = st.chat_input("원하는 영화를 찾지 못했나요? UniCRS와 대화해보세요!")
    if user_message:
        response = get_unicrs_response(
            user_message, st.session_state.dialog, st.session_state.pipeline
        )
        st.session_state.conversations.append({"role": "user", "content": user_message})
        st.session_state.conversations.append(
            {"role": "assistant", "content": response}
        )

    # -------------------------------------------------------
    # (D) 대화 표시
    #   - Like/Dislike는 한 번 누르면 둘 다 안 보이게
    # -------------------------------------------------------
    for idx, msg in enumerate(st.session_state.conversations):
        if msg["role"] == "user":
            # (기본) 왼쪽에 표시
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            # 시스템(assistant)은 오른쪽
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["content"])
                # 좋아요/싫어요 이미 제출됐는지 여부
                if not st.session_state.feedback_submitted.get(idx, False):
                    like_col, unlike_col = st.columns([0.12, 0.12], gap="small")
                    with like_col:
                        if st.button("👍", key=f"like_{idx}"):
                            st.toast("좋아요를 서버에 기록했습니다.")
                            st.session_state.feedback_submitted[idx] = True
                    with unlike_col:
                        if st.button("👎", key=f"unlike_{idx}"):
                            st.toast("별로라는 의견을 서버에 기록했습니다.")
                            st.session_state.feedback_submitted[idx] = True


if __name__ == "__main__":
    main()
