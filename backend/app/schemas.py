from pydantic import BaseModel


class SignUpRequest(BaseModel):
    reviewer_id: str
    password: str
    name: str


class SignInRequest(BaseModel):
    reviewer_id: str
    password: str


class DetailPredictRequest(BaseModel):
    item_id: str


# for test
class MainPredictRequest(BaseModel):
    user_id: str
