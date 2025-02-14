from fastapi import APIRouter, Depends

from backend.schemas import PredictRequest
from backend.services import get_prediction

router = APIRouter()


@router.post("/predict-main")
def predict(
    request: PredictRequest, dependencies: dict = Depends(lambda: router.lifespan)
):
    return get_prediction(
        request.user_id, dependencies["model_manager"], dependencies["user"]
    )
