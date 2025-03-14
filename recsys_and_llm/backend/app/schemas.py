from typing import Any, Optional

from pydantic import BaseModel


# 요청 schema
class SignUpRequest(BaseModel):
    reviewer_id: str
    password: str
    name: str


class SignInRequest(BaseModel):
    reviewer_id: str
    password: str


class DetailPredictRequest(BaseModel):
    reviewer_id: str
    item_id: str


class ReviewPostRequest(BaseModel):
    item_id: str
    reviewer_id: str
    review: str
    rating: str


# for test
class MainPredictRequest(BaseModel):
    reviewer_id: str


# 응답 schema
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None  # 성공 시 추가 데이터 포함 (Optional)
