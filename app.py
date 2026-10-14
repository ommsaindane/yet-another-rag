from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.startup import initialize_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_graph(app)
    yield


app = FastAPI(
    title="Yet Another RAG",
    description="Corrective RAG (CRAG) API with Hybrid Search, Cross-Encoder Reranking, and Incremental Indexing.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)