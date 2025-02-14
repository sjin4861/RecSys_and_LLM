import uvicorn
from app.routers import router
from fastapi import FastAPI

app = FastAPI()

# 라우터 등록
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
