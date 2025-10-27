"""
LLM-Assisted Chunker - Main Application

This is the main entry point for the LLM-Assisted Chunker application.
It orchestrates the workflow for document chunking with multiple chunking strategies.

New Workflow:
1. Get input file and extract text
2. Detect file type and size
3. Get LLM suggestion for optimal chunking strategy
4. Let user choose chunking strategy
5. Perform chosen chunking
6. Save results
"""

import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Import our modules
import sys
from pathlib import Path

# Add utils directory to path
utils_path = Path(__file__).parent.parent / 'utils'
sys.path.insert(0, str(utils_path))

from utils.text_utils import extract_text, count_tokens, get_text_excerpt, get_file_info, SUPPORTED_EXTENSIONS
from utils.llm_utils import suggest_chunking_strategy
from chunking_strategies import (
    sentence_based_chunking,
    paragraph_chunking,
    section_based_chunking,
    semantic_based_chunking,
    CHUNKING_STRATEGIES
)


# Configure logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'chunking.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)

# Fix console encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


def main():
    """
    Main application workflow.
    
    New Workflow:
    1. Get file path from user
    2. Extract file information (type, size)
    3. Extract text from document
    4. Get LLM recommendation for chunking strategy
    5. Let user choose chunking strategy
    6. Perform chosen chunking
    7. Save results
    """
    logger.info("=" * 60)
    logger.info("LLM-Assisted Chunker - Starting Application")
    logger.info("=" * 60)
    
    try:
        # Step 1: Get file path from user
        file_path = get_file_path_from_user()
        
        # Step 2: Extract file information
        logger.info("Step 1: Analyzing file information")
        file_info = get_file_info(file_path)
        logger.info(f"✓ File Type: {file_info['file_type']}")
        logger.info(f"✓ File Size: {file_info['size_kb']} KB")
        
        # Step 3: Extract text from document
        logger.info("Step 2: Extracting text from document")
        text = extract_text(file_path)
        logger.info(f"✓ Text extraction complete: {len(text)} characters")
        
        # Count tokens
        token_count = count_tokens(text)
        logger.info(f"✓ Token count: {token_count} tokens")
        
        # Step 4: Get LLM suggestion for chunking strategy
        logger.info("Step 3: Getting LLM recommendation for chunking strategy")
        excerpt = get_text_excerpt(text, max_chars=1000)
        suggestion = suggest_chunking_strategy(
            file_info['file_type'],
            file_info['size_kb'],
            excerpt
        )
        
        logger.info(f"✓ LLM Recommendation: {suggestion.get('strategy_name', 'Unknown')}")
        logger.info(f"✓ Reasoning: {suggestion.get('reasoning', 'No reasoning provided')}")
        
        # Step 5: Let user choose chunking strategy
        logger.info("Step 4: User selecting chunking strategy")
        chosen_strategy = get_user_chunking_choice(suggestion)
        
        # Step 6: Perform chosen chunking
        logger.info(f"Step 5: Performing {CHUNKING_STRATEGIES[chosen_strategy][0]}")
        chunks = perform_chunking(text, chosen_strategy)
        logger.info(f"✓ Chunking complete: {len(chunks)} chunks created")
        
        # Step 7: Display chunk summary
        display_chunk_summary(chunks)
        
        # Step 8: Save chunks to file
        output_file = save_chunks_to_file(chunks, file_path, chosen_strategy)
        logger.info(f"✓ Chunks saved to: {output_file}")
        
        # Final summary
        logger.info("=" * 60)
        logger.info("CHUNKING COMPLETE - Summary:")
        logger.info(f"  Input file: {file_path}")
        logger.info(f"  File type: {file_info['file_type']}")
        logger.info(f"  Document size: {token_count} tokens ({len(text)} characters)")
        logger.info(f"  Chunking strategy: {CHUNKING_STRATEGIES[chosen_strategy][0]}")
        logger.info(f"  Number of chunks: {len(chunks)}")
        logger.info(f"  Output file: {output_file}")
        logger.info("=" * 60)
        
        print(f"\n✓ SUCCESS! Created {len(chunks)} chunks using {CHUNKING_STRATEGIES[chosen_strategy][0]}.")
        print(f"✓ Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\nApplication interrupted by user.")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print(f"\n✗ ERROR: {str(e)}")
        sys.exit(1)


def get_file_path_from_user() -> str:
    """
    Prompt user for document file path and validate it.
    
    Returns:
        str: Validated file path
    """
    print("\n" + "=" * 60)
    print("LLM-Driven Chunker")
    print("=" * 60)
    print(f"\nSupported file formats: {', '.join(SUPPORTED_EXTENSIONS)}")
    print()
    
    while True:
        file_path = input("Enter document path: ").strip()
        
        # Remove quotes if user wrapped path in quotes
        file_path = file_path.strip('"').strip("'")
        
        if not file_path:
            print("✗ Error: File path cannot be empty. Please try again.")
            continue
        
        path = Path(file_path)
        
        if not path.exists():
            print(f"✗ Error: File not found: {file_path}")
            print("  Please check the path and try again.")
            continue
        
        if not path.is_file():
            print(f"✗ Error: Path is not a file: {file_path}")
            continue
        
        # Check file extension
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"✗ Error: Unsupported file format: {ext}")
            print(f"  Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
            continue
        
        logger.info(f"File path accepted: {file_path}")
        return str(path.absolute())


def get_user_chunking_choice(suggestion: Dict[str, Any]) -> str:
    """
    Prompt user to choose a chunking strategy.
    
    Args:
        suggestion (Dict[str, Any]): LLM's suggested strategy
        
    Returns:
        str: User's chosen strategy key ("1", "2", "3", or "4")
    """
    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY SELECTION")
    print("=" * 60)
    
    # Display LLM recommendation
    print("\n🤖 LLM Recommendation:")
    print(f"   Strategy: {suggestion.get('strategy_name', 'Unknown')}")
    print(f"   Reason: {suggestion.get('reasoning', 'No reasoning provided')}")
    
    # Display all options
    print("\n📋 Available Chunking Strategies:")
    for key, (name, _) in CHUNKING_STRATEGIES.items():
        recommended = " ⭐ [RECOMMENDED]" if key == suggestion.get('recommended_strategy') else ""
        print(f"   {key}. {name}{recommended}")
    
    print()
    
    # Get user choice
    while True:
        choice = input("Select chunking strategy (1-4): ").strip()
        
        if choice in CHUNKING_STRATEGIES:
            logger.info(f"User selected strategy {choice}: {CHUNKING_STRATEGIES[choice][0]}")
            return choice
        else:
            print("✗ Invalid choice. Please enter a number between 1 and 4.")


def perform_chunking(text: str, strategy_key: str) -> List[str]:
    """
    Perform chunking using the selected strategy.
    
    Args:
        text (str): The text to chunk
        strategy_key (str): The strategy key ("1", "2", "3", or "4")
        
    Returns:
        List[str]: List of text chunks
    """
    strategy_name, strategy_function = CHUNKING_STRATEGIES[strategy_key]
    
    logger.info(f"Performing {strategy_name}")
    
    try:
        # Call the appropriate chunking function
        chunks = strategy_function(text)
        return chunks
    except Exception as e:
        logger.error(f"Error during chunking: {str(e)}", exc_info=True)
        raise


def display_chunk_summary(chunks: List[str]) -> None:
    """
    Display a summary of the chunks to the user.
    
    Args:
        chunks (List[str]): List of text chunks
    """
    print("\n" + "-" * 60)
    print("CHUNK SUMMARY")
    print("-" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        tokens = count_tokens(chunk)
        preview = chunk[:100].replace('\n', ' ')
        if len(chunk) > 100:
            preview += "..."
        
        print(f"\nChunk {i}:")
        print(f"  Tokens: {tokens}")
        print(f"  Characters: {len(chunk)}")
        print(f"  Preview: {preview}")
    
    print("-" * 60)


def save_chunks_to_file(chunks: List[str], input_file_path: str, strategy_key: str) -> str:
    """
    Save chunks to a JSON file.
    
    Args:
        chunks (List[str]): List of text chunks
        input_file_path (str): Original input file path (for naming output)
        strategy_key (str): The strategy key used for chunking
        
    Returns:
        str: Path to output file
    """
    logger.info("Saving chunks to file...")
    
    # Create output filename based on input filename
    input_path = Path(input_file_path)
    output_filename = f"{input_path.stem}_chunks.json"
    
    # Save to output directory (create if doesn't exist)
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename
    
    # Prepare output data
    output_data = {
        "source_file": str(input_path.absolute()),
        "chunking_strategy": CHUNKING_STRATEGIES[strategy_key][0],
        "num_chunks": len(chunks),
        "chunks": []
    }
    
    for i, chunk in enumerate(chunks, 1):
        chunk_data = {
            "chunk_id": i,
            "text": chunk,
            "token_count": count_tokens(chunk),
            "char_count": len(chunk)
        }
        output_data["chunks"].append(chunk_data)
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Chunks saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    main()
