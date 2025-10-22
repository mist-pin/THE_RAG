from fastapi import FastAPI
from vector_store.api.handler import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vector Store API")

# Allow CORS for frontend/local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Vector Store API running 🚀"}
