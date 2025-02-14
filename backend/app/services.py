def get_prediction(user_id: int, model_manager, user_collection):
    user_data = user_collection.find_one({"_id": user_id})

    if not user_data:
        return {"error": "존재하지 않는 유저입니다."}

    seq = [item["itemnum"] for item in user_data.get("items", [])]

    if model_manager is None:
        return {"error": "모델이 로드되지 않았습니다."}

    return model_manager.inference(user_id, seq)
