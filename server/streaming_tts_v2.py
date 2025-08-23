"""
Streaming TTS implementation that properly integrates with Pipecat pipeline.
This version focuses on reducing perceived latency while maintaining compatibility.
"""

from typing import Optional, List, Union, Any
import asyncio
import re
from loguru import logger


def create_streaming_tts_wrapper(base_tts_service, chunk_size: int = 12, max_delay: float = 0.05):
    """
    Create a streaming TTS service by modifying the process_frame method of the base service.
    This approach maintains full compatibility with the pipeline while adding streaming capabilities.
    """
    
    # Store the original process_frame method
    original_process_frame = base_tts_service.process_frame
    
    async def streaming_process_frame(self, frame):
        """Enhanced process_frame that adds streaming capabilities."""
        try:
            # Check if this frame has text that we should stream
            if hasattr(frame, 'text') and frame.text and len(frame.text.split()) > chunk_size:
                logger.debug(f"Streaming TTS: Processing long text ({len(frame.text.split())} words)")
                
                # Split text into manageable chunks
                chunks = _split_text_into_chunks(frame.text, chunk_size)
                
                if len(chunks) > 1:
                    # Process first chunk immediately
                    first_chunk_frame = _create_frame_copy(frame)
                    first_chunk_frame.text = chunks[0]
                    
                    # Process the first chunk using the original method
                    result = await original_process_frame(self, first_chunk_frame)
                    
                    # Start background processing for remaining chunks
                    if len(chunks) > 1:
                        asyncio.create_task(_process_remaining_chunks(
                            base_tts_service, chunks[1:], frame, original_process_frame, max_delay
                        ))
                    
                    return result
            
            # For short text or non-text frames, use the original method
            return await original_process_frame(self, frame)
            
        except Exception as e:
            logger.error(f"Error in streaming TTS process_frame: {e}")
            # Fall back to original processing on error
            return await original_process_frame(self, frame)
    
    # Replace the process_frame method with our enhanced version
    base_tts_service.process_frame = streaming_process_frame
    
    logger.info(f"Enhanced TTS service with streaming capabilities (chunk_size={chunk_size})")
    return base_tts_service


def _split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks for streaming."""
    # First try to split by sentences for more natural breaks
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # If we have multiple sentences and they're reasonable size, use them
    if len(sentences) > 1:
        # Check if sentences are reasonably sized
        reasonable_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) <= chunk_size * 2:  # Allow sentences up to 2x chunk size
                reasonable_sentences.append(sentence)
            else:
                # Split long sentences by words
                reasonable_sentences.extend(_split_by_words(sentence, chunk_size))
        
        if reasonable_sentences:
            return reasonable_sentences
    
    # Fall back to word-based splitting
    return _split_by_words(text, chunk_size)


def _split_by_words(text: str, chunk_size: int) -> List[str]:
    """Split text by words into chunks."""
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(words):
        # Get chunk words
        end_idx = min(i + chunk_size, len(words))
        chunk_words = words[i:end_idx]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        i = end_idx
    
    return chunks


def _create_frame_copy(original_frame):
    """Create a copy of the frame for chunk processing."""
    # Create a simple copy that preserves the important attributes
    frame_copy = type(original_frame.__class__.__name__, (), {})
    
    # Copy all attributes from the original frame
    for attr in dir(original_frame):
        if not attr.startswith('_') and hasattr(original_frame, attr):
            try:
                value = getattr(original_frame, attr)
                if not callable(value):  # Don't copy methods
                    setattr(frame_copy, attr, value)
            except:
                pass  # Skip attributes that can't be copied
    
    return frame_copy


async def _process_remaining_chunks(
    tts_service, chunks: List[str], original_frame, process_frame_method, delay: float
):
    """Process remaining text chunks in the background."""
    for chunk in chunks:
        try:
            # Create a frame for this chunk
            chunk_frame = _create_frame_copy(original_frame)
            chunk_frame.text = chunk
            
            # Small delay to allow first chunk to start playing
            await asyncio.sleep(delay)
            
            # Process the chunk (need to pass 'self' which is the tts_service)
            result = await process_frame_method(tts_service, chunk_frame)
            
            # The result will be automatically handled by the pipeline
            logger.debug(f"Streaming TTS: Processed chunk: {chunk[:30]}...")
            
        except Exception as e:
            logger.error(f"Error processing background chunk: {e}")
