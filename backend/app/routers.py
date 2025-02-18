import os
import sys

from fastapi import APIRouter, Depends

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app.dependencies import get_dependencies
from backend.app.schemas import *
from backend.app.services import *

router = APIRouter()


@router.post("/sign-up")
def predict(request: SignUpRequest, dependencies: dict = Depends(get_dependencies)):
    return sign_up(request, dependencies["user"])


@router.post("/sign-in")
def predict(request: SignInRequest, dependencies: dict = Depends(get_dependencies)):
    return sign_in(
        request,
        dependencies["model_manager"],
        dependencies["user"],
        dependencies["item"],
    )


@router.post("/detail")
def predict(
    request: DetailPredictRequest, dependencies: dict = Depends(get_dependencies)
):
    return detail_prediction(
        request,
        dependencies["model_manager"],
        dependencies["item"],
        dependencies["review"],
    )


@router.post("/review")
def predict(request: ReviewPostRequest, dependencies: dict = Depends(get_dependencies)):
    return review_post(
        request,
        dependencies["user"],
        dependencies["item"],
        dependencies["review"],
    )


# for test
@router.post("/main")
def predict(
    request: MainPredictRequest, dependencies: dict = Depends(get_dependencies)
):
    return main_prediction(
        request,
        dependencies["model_manager"],
        dependencies["user"],
        dependencies["item"],
    )
