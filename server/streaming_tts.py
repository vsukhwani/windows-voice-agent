from typing import Optional, List, Union, Any, Dict
import asyncio
import re
from loguru import logger


class StreamingTTSService:
    """
    A TTS service that streams audio in chunks for lower latency.
    This acts as a wrapper around the base TTS service.
    """

    def __init__(
        self,
        tts_service: Any,
        chunk_size: int = 15,
        overlap: int = 2,
        max_delay: float = 0.1,
    ):
        """
        Initialize the streaming TTS service.
        
        Args:
            tts_service: The base TTS service to use for synthesis
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between chunks
            max_delay: Maximum delay between chunks in seconds
        """
        self.tts_service = tts_service
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_delay = max_delay
        self.next_processor = None
        
        # Copy attributes from the base TTS service to make this compatible
        if hasattr(tts_service, '__dict__'):
            for attr, value in tts_service.__dict__.items():
                if not hasattr(self, attr):
                    setattr(self, attr, value)
        
        logger.info(f"Streaming TTS initialized with chunk_size={chunk_size}, overlap={overlap}")

    def __getattr__(self, name):
        """Delegate any missing attributes/methods to the base TTS service."""
        return getattr(self.tts_service, name)

    def link(self, next_processor):
        """Link this processor to the next one in the pipeline."""
        logger.debug(f"Linking StreamingTTSService to {next_processor.__class__.__name__}")
        # Link the base TTS service instead of ourselves to maintain compatibility
        return self.tts_service.link(next_processor)
        
    async def process_frame(self, frame):
        """Process a frame by delegating to the base TTS service."""
        try:
            # For now, we'll just delegate to the base TTS service
            # This ensures compatibility while we can add streaming later
            return await self.tts_service.process_frame(frame)
            
        except Exception as e:
            logger.error(f"Error in StreamingTTSService.process_frame: {e}")
            # On error, just return the original frame
            return frame
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for streaming."""
        # First try to split by sentences for more natural breaks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If we have multiple sentences, use them as chunks
        if len(sentences) > 1:
            return sentences
        
        # If no sentence breaks, split by words
        words = text.split()
        
        # If text is shorter than chunk size, return it as is
        if len(words) <= self.chunk_size:
            return [text]
            
        chunks = []
        i = 0
        
        while i < len(words):
            # Get chunk words
            chunk_words = words[i:i + self.chunk_size]
            # Create chunk text
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            # Move to next chunk position, accounting for overlap
            i += max(1, self.chunk_size - self.overlap)
            
        return chunks

    async def process_frame(self, frame):
        """Process a frame by streaming TTS audio in chunks."""
        
        # If it's already a TTSAudioFrame with audio, pass it through
        if hasattr(frame, 'audio') and frame.audio:
            return frame
            
        # If it's an assistant response, process it for streaming
        if hasattr(frame, 'text') and frame.text:
            text = frame.text
            logger.debug(f"Streaming TTS processing text: {text[:50]}...")
            
            # Split text into chunks
            chunks = self._split_into_chunks(text)
            logger.debug(f"Split into {len(chunks)} chunks")
            
            # Process first chunk immediately and return
            if chunks:
                first_chunk = chunks[0]
                audio = await self._synthesize(first_chunk)
                
                # Start background task to process remaining chunks
                if len(chunks) > 1:
                    source_id = frame.source_id if hasattr(frame, 'source_id') else None
                    asyncio.create_task(self._process_remaining_chunks(chunks[1:], source_id))
                
                # Create audio frame by copying attributes from the original frame
                frame.audio = audio
                return frame
        
        # For other frame types, pass to base TTS service
        try:
            return await self.tts_service.process_frame(frame)
        except Exception as e:
            logger.error(f"Error in base TTS service: {e}")
            return frame

    async def _process_remaining_chunks(self, chunks: List[str], source_id=None):
        """Process remaining chunks in the background and queue them."""
        for chunk in chunks:
            audio = await self._synthesize(chunk)
            if audio:
                # Create a frame with audio
                # We need to use the same class that the rest of the pipeline expects
                # This creates a simple dict-like object that can be used like a frame
                frame = type('AudioFrame', (), {
                    'audio': audio,
                    'text': chunk,
                    'source_id': source_id
                })
                
                # Queue the frame for output
                await self.emit_frame(frame)
                # Small delay to allow for natural speech cadence
                await asyncio.sleep(self.max_delay)
                
    async def emit_frame(self, frame):
        """Emit a frame to the next processor in the pipeline."""
        # If a callback is registered, use it to emit the frame
        if hasattr(self, 'emit_frame_callback') and self.emit_frame_callback:
            await self.emit_frame_callback(frame)
        else:
            logger.warning("No emit_frame_callback registered. Cannot emit frame.")

    async def _synthesize(self, text: str) -> Union[bytes, None]:
        """Synthesize a text chunk into audio."""
        try:
            # Create a simple object with text attribute
            frame = type('TextFrame', (), {'text': text})
            
            # Process the frame using the base TTS service
            result = await self.tts_service.process_frame(frame)
            
            # Extract audio from the result
            if result and hasattr(result, 'audio'):
                return result.audio
            return None
        except Exception as e:
            logger.error(f"Error synthesizing chunk: {e}")
            return None

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for streaming."""
        # First try to split by sentences for more natural breaks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If we have multiple sentences, use them as chunks
        if len(sentences) > 1:
            return sentences
        
        # If no sentence breaks, split by words
        words = text.split()
        
        # If text is shorter than chunk size, return it as is
        if len(words) <= self.chunk_size:
            return [text]
            
        chunks = []
        i = 0
        
        while i < len(words):
            # Get chunk words
            chunk_words = words[i:i + self.chunk_size]
            # Create chunk text
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            # Move to next chunk position, accounting for overlap
            i += max(1, self.chunk_size - self.overlap)
            
        return chunks
