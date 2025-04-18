import json

def convert_redial_to_llamafactory(example):
    movie_dict = example["movieMentions"]
    messages = example["messages"]

    dialogue = []
    recommendations = []

    for msg in messages:
        text = msg["text"]
        # @movie_id → actual movie title
        for mid, title in movie_dict.items():
            if title is not None:  # title이 None이 아닌 경우만 처리
                text = text.replace(f"@{mid}", title)
            else:
                text = text.replace(f"@{mid}", f"movie_{mid}")  # None인 경우 대체 텍스트 사용

        role = "User" if msg["senderWorkerId"] == example["initiatorWorkerId"] else "System"
        dialogue.append(f"{role}: {text}")

        if role == "System" and any(f"@{mid}" in msg["text"] for mid in movie_dict):
            for mid in movie_dict:
                if f"@{mid}" in msg["text"] and movie_dict[mid] is not None:
                    recommendations.append(movie_dict[mid])

    # 대화가 없거나 System 응답이 없는 경우 처리
    if not dialogue or not any(line.startswith("System:") for line in dialogue):
        return None

    # 마지막 System 응답 추출
    last_output = [line for line in dialogue if line.startswith("System:")][-1].replace("System: ", "")
    
    # 추천된 영화 리스트가 있으면 그것을 output으로 사용
    if recommendations:
        movie_recommendations = ", ".join(list(set(recommendations)))  # 중복 제거
        output = f"Recommended movies: {movie_recommendations}"
    else:
        # 추천 영화가 없으면 마지막 System 응답을 사용
        output = last_output

    return {
        "instruction": "Based on the conversation, recommend movies to the user.",
        "input": "\n".join(dialogue[:-1]),
        "output": output
    }

def process_dataset(input_file, output_file):
    """전체 데이터셋을 처리하여 Llama Factory 형식으로 변환"""
    print(f"Processing {input_file}...")
    
    # 입력 데이터 로드
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} conversations")
    
    # 데이터 변환
    converted_data = []
    for example in data:
        try:
            result = convert_redial_to_llamafactory(example)
            if result:
                converted_data.append(result)
        except Exception as e:
            print(f"Error processing an example: {e}")
            continue
    
    print(f"Converted {len(converted_data)} valid conversations")
    
    # 결과 저장
    with open(output_file, "w") as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # 훈련 데이터 처리
    process_dataset("train_data.jsonl", "train_data_llamafactory.jsonl")
    
    # 테스트 데이터 처리
    process_dataset("test_data.jsonl", "test_data_llamafactory.jsonl")
    
    print("Processing complete!")
