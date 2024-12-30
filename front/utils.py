import streamlit as st


def init_session_state():
    st.session_state.user_id = ""
    st.session_state.rec_results = None
