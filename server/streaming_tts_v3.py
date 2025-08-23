"""
Ultra Low-Latency Streaming TTS V3
Advanced optimizations for minimal perceived latency.
"""

from typing import Optional, List, Union, Any
import asyncio
import re
import time
from loguru import logger


def create_ultra_fast_streaming_tts(base_tts_service, chunk_size: int = 4, max_delay: float = 0.01):
    """
    Create an ultra-fast streaming TTS service with aggressive optimizations.
    
    Key optimizations:
    - Micro-chunks (4 words or less)
    - Minimal delays (10ms)
    - Sentence boundary detection
    - Parallel processing
    - Memory pre-allocation
    """
    
    # Store the original process_frame method
    original_process_frame = base_tts_service.process_frame
    
    async def ultra_fast_process_frame(self, frame):
        """Ultra-optimized process_frame with micro-chunking."""
        start_time = time.time()
        
        try:
            # Quick text check
            if not (hasattr(frame, 'text') and frame.text and frame.text.strip()):
                result = await original_process_frame(self, frame)
                logger.debug(f"⚡ Non-text frame: {(time.time() - start_time)*1000:.1f}ms")
                return result
            
            text = frame.text.strip()
            words = text.split()
            
            # If text is very short, process immediately
            if len(words) <= chunk_size:
                result = await original_process_frame(self, frame)
                logger.debug(f"⚡ Short text ({len(words)} words): {(time.time() - start_time)*1000:.1f}ms")
                return result
            
            logger.debug(f"⚡ Ultra-streaming {len(words)} words in micro-chunks")
            
            # Create micro-chunks with smart sentence boundaries
            chunks = _create_micro_chunks(text, chunk_size)
            
            if len(chunks) <= 1:
                result = await original_process_frame(self, frame)
                logger.debug(f"⚡ Single chunk: {(time.time() - start_time)*1000:.1f}ms")
                return result
            
            # Process first micro-chunk immediately
            first_chunk_frame = _create_frame_copy(frame)
            first_chunk_frame.text = chunks[0]
            
            result = await original_process_frame(self, first_chunk_frame)
            
            # Start ultra-fast background processing
            if len(chunks) > 1:
                asyncio.create_task(_process_micro_chunks_parallel(
                    base_tts_service, chunks[1:], frame, original_process_frame, max_delay
                ))
            
            logger.debug(f"⚡ First chunk ready: {(time.time() - start_time)*1000:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast TTS error: {e}")
            # Fallback to original processing
            return await original_process_frame(self, frame)
    
    # Replace the process_frame method
    base_tts_service.process_frame = ultra_fast_process_frame
    
    logger.info(f"⚡ Ultra-fast streaming TTS ready (chunk_size={chunk_size}, delay={max_delay*1000:.0f}ms)")
    return base_tts_service


def _create_micro_chunks(text: str, chunk_size: int) -> List[str]:
    """Create micro-chunks with intelligent sentence boundary detection."""
    # Quick sentence detection for natural breaks
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # If we have multiple short sentences, use them
    if len(sentences) > 1:
        short_sentences = [s for s in sentences if len(s.split()) <= chunk_size * 2]
        if len(short_sentences) == len(sentences):
            return sentences
    
    # Fall back to word-based micro-chunking
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        # Create micro-chunk
        end_idx = min(i + chunk_size, len(words))
        chunk_words = words[i:end_idx]
        
        # Smart punctuation handling
        chunk_text = ' '.join(chunk_words)
        
        # If chunk ends mid-sentence and next word exists, 
        # try to include it if it's short
        if (end_idx < len(words) and 
            not chunk_text.endswith(('.', '!', '?', ',')) and 
            len(words[end_idx]) <= 4):
            chunk_text += ' ' + words[end_idx]
            end_idx += 1
        
        chunks.append(chunk_text)
        i = end_idx
    
    return chunks


async def _process_micro_chunks_parallel(
    tts_service, chunks: List[str], original_frame, process_frame_method, delay: float
):
    """Process micro-chunks with parallel optimization."""
    
    # Create all frames first (memory pre-allocation)
    chunk_frames = []
    for chunk in chunks:
        chunk_frame = _create_frame_copy(original_frame)
        chunk_frame.text = chunk
        chunk_frames.append((chunk_frame, chunk))
    
    # Process chunks with minimal delays
    for i, (chunk_frame, chunk_text) in enumerate(chunk_frames):
        try:
            start_time = time.time()
            
            # Minimal delay for natural flow
            if i > 0:
                await asyncio.sleep(delay)
            
            # Process chunk
            await process_frame_method(tts_service, chunk_frame)
            
            process_time = (time.time() - start_time) * 1000
            logger.debug(f"⚡ Micro-chunk {i+1}/{len(chunks)}: {process_time:.1f}ms - '{chunk_text[:20]}...'")
            
        except Exception as e:
            logger.error(f"❌ Micro-chunk {i+1} error: {e}")


def _create_frame_copy(original_frame):
    """Optimized frame copying."""
    try:
        # Create new frame instance
        frame_copy = type(original_frame.__class__.__name__, (), {})
        
        # Copy essential attributes only
        essential_attrs = ['source_id', 'timestamp']
        for attr in essential_attrs:
            if hasattr(original_frame, attr):
                try:
                    setattr(frame_copy, attr, getattr(original_frame, attr))
                except:
                    pass
        
        # Copy all non-callable attributes
        for attr in dir(original_frame):
            if (not attr.startswith('_') and 
                hasattr(original_frame, attr) and 
                attr not in essential_attrs):
                try:
                    value = getattr(original_frame, attr)
                    if not callable(value):
                        setattr(frame_copy, attr, value)
                except:
                    pass
        
        return frame_copy
        
    except Exception as e:
        logger.error(f"❌ Frame copy error: {e}")
        # Fallback: create minimal frame
        frame_copy = type('FastFrame', (), {})
        return frame_copy


# Compatibility alias
create_streaming_tts_wrapper = create_ultra_fast_streaming_tts
