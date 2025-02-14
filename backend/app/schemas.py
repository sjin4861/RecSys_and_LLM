from pydantic import BaseModel


class PredictRequest(BaseModel):
    user_id: int
