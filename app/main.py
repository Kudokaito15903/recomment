from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.api import router

app = FastAPI(title="Realtime AI Recommender", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/metrics", make_asgi_app())


@app.get("/")
async def root():
    return {
        "service": "realtime-ai-recommender",
        "docs": "/docs",
        "metrics": "/metrics",
    }


def get_app():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

