import chromadb
from chromadb.config import Settings
from vector_store.core.logic_helper import get_embeddings

# Persistent storage for Chroma
PERSIST_DIR = "./vector_store_data"
chroma_client = chromadb.Client(Settings(is_persistent=True, persist_directory=PERSIST_DIR))

def add_chunks_to_db(payload):
    collection_name = payload.get("collection_name", "default_collection")
    chunks = payload.get("chunks", [])
    ids = [f"id_{i}" for i in range(len(chunks))]

    embeddings = get_embeddings(chunks)
    collection = chroma_client.get_or_create_collection(collection_name)
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return {"message": f"Added {len(chunks)} chunks to {collection_name}"}

def query_similar_chunks(payload):
    collection_name = payload.get("collection_name", "default_collection")
    query_text = payload.get("query_text", "")

    collection = chroma_client.get_or_create_collection(collection_name)
    query_embedding = get_embeddings([query_text])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return {"results": results}

def update_existing_chunk(payload):
    collection_name = payload.get("collection_name", "default_collection")
    chunk_id = payload.get("chunk_id")
    new_text = payload.get("new_text")

    collection = chroma_client.get_or_create_collection(collection_name)
    new_embedding = get_embeddings([new_text])[0]
    collection.update(ids=[chunk_id], documents=[new_text], embeddings=[new_embedding])
    return {"message": f"Chunk {chunk_id} updated."}

def delete_collection_data(collection_name):
    chroma_client.delete_collection(name=collection_name)
    return {"message": f"Collection '{collection_name}' deleted successfully."}
