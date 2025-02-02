# app/components/conversation_manager.py
import json
import os

import streamlit as st


def save_conversation(title):
    """
    대화를 저장하는 함수
    """
    if title.strip():
        st.session_state.saved_conversations[title] = {
            "conversations": st.session_state.conversations.copy(),
            "dialog": st.session_state.dialog.copy(),
        }
        return f"'{title}' 대화를 저장했습니다!"
    return "세션 제목이 비어있어 저장을 취소했습니다."


def load_conversation(title):
    """
    저장된 대화 불러오기
    """
    stored_data = st.session_state.saved_conversations[title]
    st.session_state.conversations = stored_data["conversations"].copy()
    st.session_state.dialog = stored_data["dialog"].copy()
    return f"'{title}' 대화를 불러왔습니다."


def persist_conversation(user_id):
    """Store the user's conversations, dialog, and saved conversations in a local JSON file."""
    data = {
        "conversations": st.session_state.conversations,
        "dialog": st.session_state.dialog,
        "saved_conversations": st.session_state.saved_conversations,  # 추가
    }
    with open(f"{user_id}_session.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def retrieve_persisted_conversation(user_id):
    """Load any previously stored conversation data for this user."""
    filename = f"{user_id}_session.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state.conversations = data.get("conversations", [])
            st.session_state.dialog = data.get("dialog", [])
            st.session_state.saved_conversations = data.get(
                "saved_conversations", {}
            )  # 추가
