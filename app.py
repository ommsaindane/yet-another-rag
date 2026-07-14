from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from api.startup import initialize_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_graph()
    yield


app = FastAPI(
    title="Yet Another RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
