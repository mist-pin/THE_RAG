"""
LangGraph Pipeline: Orchestrates the entire RAG flow
User Input → Feature Extraction → Vector DB → Answer Generation
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import requests
import json
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from root directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class GraphState(TypedDict):
    """State shared across all nodes in the graph"""
    user_input: str
    task: str
    features: List[str]
    retrieved_context: str
    final_answer: str
    error: str


def call_groq_api(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq API with LLM model"""
    try:
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY not found in environment variables. Set it in .env file")
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        raise Exception(f"Groq API timeout - request took too long")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to Groq API. Check your internet connection.")
    except Exception as e:
        raise Exception(f"Groq API call failed: {str(e)}")


def extract_features_node(state: GraphState) -> GraphState:
    """Node 1: Extract task and features from user input using LLM"""
    user_input = state["user_input"]
    
    prompt = f"""Analyze the following user input and extract:
1. The main task (a concise description)
2. The features (key entities, dates, names, etc.)
NOTE: Implement Semantic Search for better feature extraction.
User input: {user_input}

Respond in JSON format:
{{
    "task": "main task description",
    "features": ["feature1", "feature2", ...]
}}"""
    
    print(" Extracting features...")
    
    try:
        # Call Groq API
        output = call_groq_api(prompt + "\n\nProvide only valid JSON, nothing else.")
        
        # Parse JSON from response
        if '{' in output and '}' in output:
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            json_str = output[json_start:json_end]
            response = json.loads(json_str)
        else:
            response = json.loads(output)
            
        state["task"] = response.get("task", "")
        state["features"] = response.get("features", [])
        print(f"Task: {state['task']}")
        print(f"Features: {state['features']}")
        
    except Exception as e:
        state["error"] = f"Feature extraction failed: {str(e)}"
        print(f"Error: {e}")
    
    return state


def retrieve_context_node(state: GraphState) -> GraphState:
    """Node 2: Query vector DB with features to get relevant context"""
    features = state["features"]
    
    print(" Retrieving context from Vector DB...")
    
    try:
        # Import using relative path
        import sys
        import os
        
        # Add parent directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from src.external_services.vector_db_client import VectorDBClient
        
        db_client = VectorDBClient()
        retrieved_chunks = db_client.search(features)
        
        # Combine chunks into context string
        state["retrieved_context"] = "\n\n".join(retrieved_chunks)
        print(f"Retrieved {len(retrieved_chunks)} chunks")
        
    except Exception as e:
        state["error"] = f"Vector DB retrieval failed: {str(e)}"
        state["retrieved_context"] = ""
        print(f" Error: {e}")
    
    return state
def generate_answer_node(state: GraphState) -> GraphState:
    """Node 3: Generate final answer using LLM with context"""
    user_query = state["user_input"]
    context = state["retrieved_context"]
    
    prompt = f"""You are an intelligent AI assistant. Use the provided context to answer the user's question accurately. If the answer is not in the context, say 'I don't have enough information to answer that.' Do not hallucinate.
User Query: {user_query}

Context:
{context}

Now provide the answer."""
    
    print(" Generating answer...")
    
    try:
        # Call Groq API
        answer = call_groq_api(prompt)
        state["final_answer"] = answer.strip()
        print(" Answer generated")
        
    except Exception as e:
        state["error"] = f"Answer generation failed: {str(e)}"
        print(f" Error: {e}")
    
    return state


def should_continue(state: GraphState) -> str:
    """Conditional edge: Check if we should continue or end"""
    if state.get("error"):
        return "error"
    return "continue"
def create_pipeline():
    """Create and compile the LangGraph pipeline"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("extract_features", extract_features_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Define edges (flow)
    workflow.set_entry_point("extract_features")
    workflow.add_edge("extract_features", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # Compile the graph
    app = workflow.compile()
    return app


def run_pipeline(user_input: str) -> dict:
    """Execute the complete pipeline"""
    app = create_pipeline()
    
    # Initial state
    initial_state = {
        "user_input": user_input,
        "task": "",
        "features": [],
        "retrieved_context": "",
        "final_answer": "",
        "error": ""
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    # Test the pipeline
    user_query = "What were the purchases made by Mr. Shashank on 18-08-2023?"
    result = run_pipeline(user_query)
    
    print(f"\n=== Pipeline Results ===")
    print(f"Task: {result['task']}")
    print(f"Features: {result['features']}")
    print(f"\nFinal Answer:\n{result['final_answer']}")