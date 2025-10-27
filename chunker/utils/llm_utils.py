"""
LLM Utilities Module

This module provides functions for interacting with Large Language Models (LLMs)
to recommend optimal chunking strategies based on document analysis.
Uses Google Gemini API.
"""

import logging
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai


# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
        genai.configure(api_key=GEMINI_API_KEY)
        MOCK_MODE = False
        logger.info(f"Gemini API initialized with model: {GEMINI_MODEL}")
    else:
        MOCK_MODE = True
        logger.warning("No valid Gemini API key found. Running in mock mode.")
        logger.warning("Please set GEMINI_API_KEY in your .env file")
except Exception as e:
    logger.warning(f"Gemini client initialization issue: {e}. Running in mock mode.")
    MOCK_MODE = True


def _strip_markdown_code_blocks(text: str) -> str:
    """
    Remove markdown code block markers and extract JSON from text.
    
    Gemini sometimes wraps JSON responses in ```json ... ``` blocks
    or includes extra text. This function extracts the JSON portion.
    
    Args:
        text (str): Text that might contain markdown code blocks or extra text
        
    Returns:
        str: Extracted JSON text
    """
    text = text.strip()
    
    # Try to find JSON within markdown code blocks first
    import re
    
    # Look for ```json ... ``` or ``` ... ``` patterns
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        return text
    
    # If no code blocks, try to find JSON object by looking for { ... }
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        text = match.group(0).strip()
        return text
    
    # If no JSON pattern found, just clean up markdown markers
    if text.startswith('```json'):
        text = text[7:].strip()
    elif text.startswith('```'):
        text = text[3:].strip()
    
    if text.endswith('```'):
        text = text[:-3].strip()
    
    return text


def suggest_chunking_strategy(file_type: str, file_size_kb: float, text_excerpt: str, model: str = None) -> Dict[str, Any]:
    """
    Ask an LLM to recommend the optimal chunking strategy based on file type and content.
    
    Args:
        file_type (str): Type of file (e.g., "PDF Document", "Markdown File")
        file_size_kb (float): Size of file in KB
        text_excerpt (str): A representative excerpt from the document
        model (str): The LLM model to use (default: from environment variable)
        
    Returns:
        Dict[str, Any]: Dictionary with recommended strategy and reasoning
    """
    if model is None:
        model = GEMINI_MODEL
    
    logger.info(f"Requesting chunking strategy suggestion for {file_type} ({file_size_kb} KB)")
    
    # Construct the prompt
    prompt = f"""You are an expert in text processing and document chunking.

Based on the following document information, recommend the BEST chunking strategy:

**File Information:**
- File Type: {file_type}
- File Size: {file_size_kb} KB

**Text Excerpt:**
{text_excerpt}

**Available Chunking Strategies:**
1. **Sentence-based chunking**: Groups text by sentences (good for fine-grained analysis)
2. **Paragraph chunking**: Groups text by paragraphs (good for general documents)
3. **Section-based chunking**: Groups text by sections/headers (good for structured documents like markdown)
4. **Semantic-based chunking**: Intelligent chunking based on meaning and context (good for complex texts)

Please analyze the file type and content, then recommend ONE strategy.

Respond in this JSON format:
{{
  "recommended_strategy": "<number 1-4>",
  "strategy_name": "<name of strategy>",
  "reasoning": "<brief explanation of why this strategy is best for this document>"
}}

Respond ONLY with the JSON object."""

    try:
        if MOCK_MODE:
            logger.warning("Running in MOCK MODE - no actual LLM call made")
            return _mock_strategy_suggestion(file_type, text_excerpt)
        
        # Call the Gemini API
        logger.debug(f"Calling Gemini API with model: {model}")
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        
        # Extract the response
        response_text = response.text.strip()
        logger.debug(f"LLM response received: {response_text}")
        
        # Clean up markdown code blocks if present
        response_text = _strip_markdown_code_blocks(response_text)
        
        # Parse the JSON response
        result = json.loads(response_text)
        
        if "recommended_strategy" not in result:
            logger.error("LLM response missing 'recommended_strategy' field")
            raise ValueError("Invalid LLM response format")
        
        logger.info(f"LLM recommended strategy: {result.get('strategy_name', 'Unknown')} (Option {result['recommended_strategy']})")
        logger.info(f"Reasoning: {result.get('reasoning', 'No reasoning provided')}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw response was: {response_text[:500]}")  # Show first 500 chars
        
        # Try to be more aggressive in extracting JSON
        try:
            import re
            # Look for anything that looks like our expected JSON structure
            json_match = re.search(r'\{\s*"recommended_strategy".*?\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                logger.info("Attempting to parse extracted JSON pattern")
                result = json.loads(json_text)
                return result
        except:
            pass
        
        logger.warning("Falling back to mock strategy suggestion")
        return _mock_strategy_suggestion(file_type, text_excerpt)
        
    except Exception as e:
        logger.error(f"Error getting LLM strategy suggestion: {str(e)}")
        logger.warning("Falling back to mock strategy suggestion")
        return _mock_strategy_suggestion(file_type, text_excerpt)


def _mock_strategy_suggestion(file_type: str, text_excerpt: str) -> Dict[str, Any]:
    """
    Mock implementation for strategy suggestion.
    
    Args:
        file_type (str): Type of file
        text_excerpt (str): Text excerpt
        
    Returns:
        Dict[str, Any]: Default strategy recommendation
    """
    logger.info("Using mock strategy suggestion")
    
    # Simple heuristic based on file type
    if "markdown" in file_type.lower() or "md" in file_type.lower():
        return {
            "recommended_strategy": "3",
            "strategy_name": "Section-based chunking",
            "reasoning": "Markdown files typically have clear section structure with headers."
        }
    elif "csv" in file_type.lower() or "table" in file_type.lower():
        return {
            "recommended_strategy": "2",
            "strategy_name": "Paragraph chunking",
            "reasoning": "Tabular data is best chunked by rows or logical groups."
        }
    else:
        return {
            "recommended_strategy": "4",
            "strategy_name": "Semantic-based chunking",
            "reasoning": "General documents benefit from intelligent semantic chunking."
        }
