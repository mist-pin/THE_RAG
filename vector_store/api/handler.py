from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from core.logic import (
    add_chunks,
    query_chunks,
    update_chunk,
    delete_collection,
    list_collections
)

router = APIRouter()

class AddChunksRequest(BaseModel):
    collection_name: str
    chunks: List[str]

class QueryRequest(BaseModel):
    collection_name: str
    query_text: str

class UpdateRequest(BaseModel):
    collection_name: str
    chunk_id: str
    new_text: str

@router.post("/add_chunks")
def add_chunks_endpoint(data: AddChunksRequest):
    return add_chunks(data.collection_name, data.chunks)

@router.post("/query_chunks")
def query_chunks_endpoint(data: QueryRequest):
    return query_chunks(data.collection_name, data.query_text)

@router.put("/update_chunk")
def update_chunk_endpoint(data: UpdateRequest):
    return update_chunk(data.collection_name, data.chunk_id, data.new_text)

@router.delete("/delete_collection/{collection_name}")
def delete_collection_endpoint(collection_name: str):
    return delete_collection(collection_name)

@router.get("/list_collections")
def list_collections_endpoint():
    return list_collections()
