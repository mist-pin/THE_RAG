from fastapi import APIRouter, HTTPException
from vector_store.core.logic import (
    add_chunks_to_db,
    query_similar_chunks,
    update_existing_chunk,
    delete_collection_data
)

router = APIRouter()

@router.post("/add_chunks")
def add_chunks_endpoint(payload: dict):
    try:
        return add_chunks_to_db(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query_chunks")
def query_chunks_endpoint(payload: dict):
    try:
        return query_similar_chunks(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update_chunk")
def update_chunk_endpoint(payload: dict):
    try:
        return update_existing_chunk(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete_collection/{collection_name}")
def delete_collection_endpoint(collection_name: str):
    try:
        return delete_collection_data(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
