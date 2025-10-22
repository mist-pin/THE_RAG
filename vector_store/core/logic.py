from chromadb import Client
from chromadb.config import Settings
from fastapi import HTTPException
from utils.constants import CHROMA_PERSIST_DIR
from core.logic_helper import get_embedding_function

embedding_func = get_embedding_function()
chroma_client = Client(Settings(persist_directory=CHROMA_PERSIST_DIR))

def add_chunks(collection_name: str, chunks: list[str]):
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return {"message": f"Added {len(chunks)} chunks to '{collection_name}'."}

def query_chunks(collection_name: str, query_text: str):
    try:
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_func
        )
        results = collection.query(query_texts=[query_text], n_results=5)
        return {"results": results}
    except Exception:
        raise HTTPException(status_code=404, detail="Collection not found")

def update_chunk(collection_name: str, chunk_id: str, new_text: str):
    collection = chroma_client.get_collection(
        name=collection_name, embedding_function=embedding_func
    )
    collection.update(ids=[chunk_id], documents=[new_text])
    return {"message": f"Chunk '{chunk_id}' updated successfully."}

def delete_collection(collection_name: str):
    collections = [c.name for c in chroma_client.list_collections()]
    if collection_name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found.")
    chroma_client.delete_collection(name=collection_name)
    return {"message": f"Collection '{collection_name}' deleted."}

def list_collections():
    return {"collections": [c.name for c in chroma_client.list_collections()]}
