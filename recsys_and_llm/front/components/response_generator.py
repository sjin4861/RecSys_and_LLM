# app/components/response_generator.py


def get_unicrs_response(user_message: str, dialog: list, pipeline) -> str:
    """
    주어진 파이프라인으로 사용자 메시지에 대한 응답 생성.
    """
    dialog.append(f"User: {user_message}")
    full_context = "<sep>".join(dialog)
    response = pipeline.response(full_context)
    dialog.append(f"{response}")
    return response
