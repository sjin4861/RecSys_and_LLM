import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI 앱 생성
app = FastAPI()


# 모델 정의 (저장된 모델 구조와 동일해야 함)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# 저장된 모델 로드
model = SimpleModel()
model.load_state_dict(torch.load("simple_model.pth"))
model.eval()  # 평가 모드로 전환


# 요청 데이터 구조 정의
class PredictionRequest(BaseModel):
    input: float  # 단일 실수 입력


# 기본 엔드포인트 정의
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend!"}


# 엔드포인트 정의
@app.post("/predict")
def predict(request: PredictionRequest):
    # 입력 데이터를 PyTorch 텐서로 변환
    input_tensor = torch.tensor([[request.input]], dtype=torch.float32)
    # 모델 예측
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return {"input": request.input, "prediction": prediction}
