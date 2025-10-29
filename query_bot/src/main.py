"""
FastAPI server for LangGraph RAG Pipeline
Provides REST API endpoints to run the pipeline via HTTP requests
"""

import sys
import os
import asyncio
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

# Robust import that works when running as a package (uvicorn src.main)
# and when running the file directly (python src/main.py).
try:
    # When imported as a package (e.g. `python -m uvicorn src.main`) use relative import
    from .pipeline import run_pipeline
except Exception:
    # Fallback for direct script execution: add src directory to sys.path and import
    src_dir = os.path.dirname(__file__)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from pipeline import run_pipeline


# Pydantic models for request/response validation
class RunPipelineRequest(BaseModel):
    """Request model for /run_pipeline endpoint"""
    user_input: str

    class Config:
        json_schema_extra = {
            "example": {
                "user_input": "What were the purchases made by Mr. Shashank on 18-08-2023?"
            }
        }


class PipelineResponse(BaseModel):
    """Response model for pipeline execution"""
    task: str
    features: list[str]
    final_answer: str
    error: str = ""


# FastAPI app instance
app = FastAPI(
    title="LangGraph RAG Pipeline API",
    description="REST API for running RAG pipeline with feature extraction and answer generation",
    version="1.0.0"
)


@app.get("/")
def root():
    """Health check / welcome endpoint"""
    return {
        "message": "LangGraph RAG Pipeline API",
        "status": "running",
        "endpoints": {
            "POST /run_pipeline": "Execute the RAG pipeline with user input",
            "GET /docs": "Interactive API documentation"
        }
    }


@app.post("/run_pipeline", response_model=Dict[str, Any])
async def run_pipeline_endpoint(request: RunPipelineRequest):
    """
    Execute the LangGraph RAG pipeline with the given user input.
    
    The pipeline:
    1. Extracts task and features from user input using LLM
    2. Retrieves relevant context from Vector DB
    3. Generates final answer using LLM with retrieved context
    
    Args:
        request: RunPipelineRequest containing user_input
        
    Returns:
        Dictionary with task, features, final_answer, and error (if any)
    """
    # Run the blocking pipeline function in a thread to avoid blocking the event loop
    result = await asyncio.to_thread(run_pipeline, request.user_input)
    return result


# CLI mode (only runs when script is executed directly, not when imported by uvicorn)
def cli_main():
    """CLI entry point for running pipeline from command line"""
    # Get user input
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
    else:
        print("Enter your query:")
        user_input = input("> ").strip()
    
    if not user_input:
        print("Error: No input provided.")
        sys.exit(1)
    
    print(f"\n📝 Processing: {user_input}\n")
    
    # Run the LangGraph pipeline
    result = run_pipeline(user_input)
    
    # Display results
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
    else:
        print(f"🎯 Task Identified: {result['task']}")
        print(f"🔍 Features Extracted: {result['features']}")
        print(f"\n💡 Answer:\n{result['final_answer']}")


if __name__ == "__main__":
    cli_main()
