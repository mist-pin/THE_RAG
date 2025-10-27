"""
Chunking Strategies Module

This module provides different chunking strategies for text:
1. Sentence-based chunking
2. Paragraph chunking
3. Section-based chunking (markdown/heading-based)
4. Semantic-based chunking
"""

import logging
import re
import sys
from typing import List
from pathlib import Path
import spacy

# Add utils directory to path
utils_path = Path(__file__).parent.parent / 'utils'
sys.path.insert(0, str(utils_path))

from utils.text_utils import count_tokens


# Configure logging
logger = logging.getLogger(__name__)

# Global spaCy model (loaded on first use)
_nlp = None


def _load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Load the spaCy language model (cached).
    
    Args:
        model_name (str): Name of the spaCy model to load
        
    Returns:
        spacy.Language: Loaded spaCy model
    """
    global _nlp
    
    if _nlp is None:
        logger.info(f"Loading spaCy model: {model_name}")
        try:
            _nlp = spacy.load(model_name)
            logger.info("spaCy model loaded successfully")
        except OSError as e:
            error_msg = (
                f"spaCy model '{model_name}' not found. "
                f"Please install it using: python -m spacy download {model_name}"
            )
            logger.error(error_msg)
            raise OSError(error_msg) from e
    
    return _nlp


def sentence_based_chunking(text: str, sentences_per_chunk: int = 2, overlap_sentences: int = 1) -> List[str]:
    """
    Chunk text by sentences.
    
    Args:
        text (str): The text to chunk
        sentences_per_chunk (int): Number of sentences per chunk (default: 5)
        overlap_sentences (int): Number of overlapping sentences between chunks (default: 1)
        
    Returns:
        List[str]: List of text chunks
    """
    logger.info(f"Starting sentence-based chunking with {sentences_per_chunk} sentences per chunk")
    
    # Load spaCy model
    nlp = _load_spacy_model()
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    logger.info(f"Extracted {len(sentences)} sentences from document")
    
    if not sentences:
        logger.warning("No sentences found in document, returning original text")
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        # Get chunk of sentences
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = " ".join(chunk_sentences)
        chunks.append(chunk_text)
        
        # Move forward by sentences_per_chunk minus overlap
        step = sentences_per_chunk - overlap_sentences
        i += max(step, 1)  # Ensure we always make progress
        
        if i >= len(sentences):
            break
    
    logger.info(f"Sentence-based chunking completed: {len(chunks)} chunks created")
    _log_chunk_statistics(chunks)
    
    return chunks


def paragraph_chunking(text: str, paragraphs_per_chunk: int = 2, overlap_paragraphs: int = 0) -> List[str]:
    """
    Chunk text by paragraphs.
    
    Args:
        text (str): The text to chunk
        paragraphs_per_chunk (int): Number of paragraphs per chunk (default: 2)
        overlap_paragraphs (int): Number of overlapping paragraphs between chunks (default: 0)
        
    Returns:
        List[str]: List of text chunks
    """
    logger.info(f"Starting paragraph-based chunking with {paragraphs_per_chunk} paragraphs per chunk")
    
    # Split by double newlines or more (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    logger.info(f"Extracted {len(paragraphs)} paragraphs from document")
    
    if not paragraphs:
        logger.warning("No paragraphs found in document, returning original text")
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(paragraphs):
        # Get chunk of paragraphs
        chunk_paragraphs = paragraphs[i:i + paragraphs_per_chunk]
        chunk_text = "\n\n".join(chunk_paragraphs)
        chunks.append(chunk_text)
        
        # Move forward by paragraphs_per_chunk minus overlap
        step = paragraphs_per_chunk - overlap_paragraphs
        i += max(step, 1)  # Ensure we always make progress
        
        if i >= len(paragraphs):
            break
    
    logger.info(f"Paragraph-based chunking completed: {len(chunks)} chunks created")
    _log_chunk_statistics(chunks)
    
    return chunks


def section_based_chunking(text: str, max_section_size: int = 2000) -> List[str]:
    """
    Chunk text by sections (markdown headers or similar structure).
    
    Args:
        text (str): The text to chunk
        max_section_size (int): Maximum tokens per section chunk (default: 2000)
        
    Returns:
        List[str]: List of text chunks
    """
    logger.info("Starting section-based chunking")
    
    # Try to detect markdown headers (# Header, ## Header, etc.)
    sections = []
    current_section = []
    
    lines = text.split('\n')
    
    for line in lines:
        # Check if line is a header (starts with #)
        if re.match(r'^#{1,6}\s+', line):
            # Save current section if it has content
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    sections.append(section_text)
                current_section = []
        
        current_section.append(line)
    
    # Add the last section
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            sections.append(section_text)
    
    # If no headers found, try alternative splitting by line breaks
    if len(sections) <= 1:
        logger.info("No markdown headers found, attempting alternative section detection")
        # Fall back to paragraph-based splitting
        sections = re.split(r'\n\s*\n', text)
        sections = [s.strip() for s in sections if s.strip()]
    
    logger.info(f"Extracted {len(sections)} sections from document")
    
    if not sections:
        logger.warning("No sections found in document, returning original text")
        return [text]
    
    # Group sections if they're too small
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for section in sections:
        section_tokens = count_tokens(section)
        
        # If this section alone exceeds max, split it
        if section_tokens > max_section_size:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split large section by paragraphs
            paragraphs = re.split(r'\n\s*\n', section)
            temp_chunk = []
            temp_tokens = 0
            
            for para in paragraphs:
                para_tokens = count_tokens(para)
                if temp_tokens + para_tokens > max_section_size and temp_chunk:
                    chunks.append('\n\n'.join(temp_chunk))
                    temp_chunk = [para]
                    temp_tokens = para_tokens
                else:
                    temp_chunk.append(para)
                    temp_tokens += para_tokens
            
            if temp_chunk:
                chunks.append('\n\n'.join(temp_chunk))
        
        # If adding this section would exceed max, save current chunk
        elif current_tokens + section_tokens > max_section_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [section]
            current_tokens = section_tokens
        else:
            current_chunk.append(section)
            current_tokens += section_tokens
    
    # Add the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    logger.info(f"Section-based chunking completed: {len(chunks)} chunks created")
    _log_chunk_statistics(chunks)
    
    return chunks


def semantic_based_chunking(text: str, chunk_size: int = 1000, overlap_ratio: float = 0.1) -> List[str]:
    """
    Chunk text into semantic segments with overlap using spaCy.
    
    This function splits text at semantic boundaries (sentences) while
    respecting the target chunk size in tokens and maintaining overlap between chunks.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Target chunk size in tokens (default: 1000)
        overlap_ratio (float): Ratio of overlap between chunks (default: 0.1 for 10%)
        
    Returns:
        List[str]: List of text chunks with overlap
    """
    logger.info(f"Starting semantic chunking with chunk_size={chunk_size}, overlap_ratio={overlap_ratio}")
    
    # Load spaCy model
    nlp = _load_spacy_model()
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    logger.info(f"Extracted {len(sentences)} sentences from document")
    
    if not sentences:
        logger.warning("No sentences found in document, returning original text")
        return [text]
    
    # Calculate overlap size in tokens
    overlap_size = int(chunk_size * overlap_ratio)
    
    # Build chunks
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # Check if adding this sentence would exceed chunk size
        if current_chunk_tokens > 0 and current_chunk_tokens + sentence_tokens > chunk_size:
            # Save current chunk
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)
            
            # Calculate overlap sentences for next chunk
            overlap_sentences = _get_overlap_sentences(current_chunk_sentences, overlap_size)
            
            # Start new chunk with overlap
            current_chunk_sentences = overlap_sentences + [sentence]
            current_chunk_tokens = sum(count_tokens(s) for s in current_chunk_sentences)
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
    
    # Add the last chunk if it has content
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append(chunk_text)
    
    logger.info(f"Semantic chunking completed: {len(chunks)} chunks created")
    _log_chunk_statistics(chunks)
    
    return chunks


def _get_overlap_sentences(sentences: List[str], overlap_size: int) -> List[str]:
    """
    Get the last few sentences that fit within the overlap size.
    
    Args:
        sentences (List[str]): List of sentences from previous chunk
        overlap_size (int): Target overlap size in tokens
        
    Returns:
        List[str]: Sentences to use as overlap in next chunk
    """
    if not sentences or overlap_size <= 0:
        return []
    
    overlap_sentences = []
    overlap_tokens = 0
    
    # Work backwards from the end of the chunk
    for sentence in reversed(sentences):
        sentence_tokens = count_tokens(sentence)
        
        if overlap_tokens + sentence_tokens <= overlap_size:
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens
        else:
            break
    
    return overlap_sentences


def _log_chunk_statistics(chunks: List[str]) -> None:
    """
    Log statistics about the created chunks.
    
    Args:
        chunks (List[str]): List of text chunks
    """
    if not chunks:
        return
    
    token_counts = [count_tokens(chunk) for chunk in chunks]
    char_counts = [len(chunk) for chunk in chunks]
    
    logger.info("Chunk Statistics:")
    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info(f"  Token counts - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts) / len(token_counts):.1f}")
    logger.info(f"  Character counts - Min: {min(char_counts)}, Max: {max(char_counts)}, Avg: {sum(char_counts) / len(char_counts):.1f}")


# Mapping of chunking strategy names to functions
CHUNKING_STRATEGIES = {
    "1": ("Sentence-based chunking", sentence_based_chunking),
    "2": ("Paragraph chunking", paragraph_chunking),
    "3": ("Section-based chunking", section_based_chunking),
    "4": ("Semantic-based chunking", semantic_based_chunking),
}
