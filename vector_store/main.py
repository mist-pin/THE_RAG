from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.handler import router

app = FastAPI(title="Vector Store API")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router)
