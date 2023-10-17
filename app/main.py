from datetime import datetime
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware


from app.api.v1.api import router as api_router
from app.core.config import settings

creation_time = datetime.utcnow()
root_router = APIRouter()
app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@root_router.get('/health')
def get_health() -> dict:
    return {
        "status": "UP",
        "name": settings.app_name,
        "creationTimeUtc": creation_time
    }


app.include_router(root_router)
app.include_router(api_router, prefix=f'/api/{settings.api_version}')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
