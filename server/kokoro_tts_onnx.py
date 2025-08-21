#
# Kokoro TTS service for Pipecat using ONNX Runtime (Windows-compatible)
#

import asyncio
import os
from typing import AsyncGenerator, Optional, Dict, Any, List

import numpy as np
from loguru import logger
import onnxruntime as ort
from ttstokenizer import IPATokenizer

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class KokoroOnnxTTSService(TTSService):
    """Kokoro TTS service implementation using ONNX Runtime.
    
    Provides text-to-speech synthesis using Kokoro models running locally
    on Windows/Linux/macOS through ONNX Runtime. Uses a separate thread
    for audio generation to avoid blocking the pipeline.
    """

    def __init__(
        self,
        *,
        model_path: str,
        voices_path: str,
        voice: str = "af_heart",
        sample_rate: int = 24000,
        providers: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize the Kokoro ONNX TTS service.

        Args:
            model_path: Path to the Kokoro ONNX model file.
            voices_path: Path to the voices binary file.
            voice: The voice to use for synthesis (default: "af_heart").
            sample_rate: Output sample rate (default: 24000).
            providers: ONNX Runtime execution providers (default: auto-detect).
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_path = model_path
        self._voices_path = voices_path
        self._voice = voice

        # Tokenizer for text-to-phoneme conversion
        self._tokenizer = IPATokenizer()

        # ONNX providers: prefer CUDA if available, fallback to CPU
        if providers is None:
            available = ort.get_available_providers()
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        else:
            self._providers = providers

        # Lazy initialization
        self._sess: Optional[ort.InferenceSession] = None
        self._input_names: List[str] = []
        self._voices: Dict[str, np.ndarray] = {}

        self._settings = {
            "model_path": model_path,
            "voices_path": voices_path,
            "voice": voice,
            "sample_rate": sample_rate,
            "providers": self._providers,
        }

    def can_generate_metrics(self) -> bool:
        return True

    def _ensure_initialized(self):
        """Initialize the ONNX model and voices if not already done."""
        if self._sess is not None:
            return

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(f"Kokoro ONNX model not found at: {self._model_path}")

        if not os.path.isfile(self._voices_path):
            raise FileNotFoundError(f"Kokoro voices file not found at: {self._voices_path}")

        logger.info(f"Loading Kokoro ONNX model: {self._model_path}")
        self._sess = ort.InferenceSession(self._model_path, providers=self._providers)
        self._input_names = [i.name for i in self._sess.get_inputs()]
        logger.info(f"Kokoro ONNX inputs: {self._input_names}; providers: {self._sess.get_providers()}")

        # Load voices mapping
        # voices-v1.0.bin is typically an NPZ file mapping {voice_name: array(shape=(N,1,256), float32)}
        voices_obj = np.load(self._voices_path, allow_pickle=True)
        if isinstance(voices_obj, np.lib.npyio.NpzFile):
            for k in voices_obj.files:
                self._voices[k] = voices_obj[k]
        elif isinstance(voices_obj, dict):
            self._voices = {k: np.asarray(v) for k, v in voices_obj.items()}
        else:
            # Fallback: single-voice .bin; name it 'default'
            arr = np.asarray(voices_obj)
            self._voices["default"] = arr

        if self._voice not in self._voices:
            # Fall back if provided voice not found
            logger.warning(f"Voice '{self._voice}' not found in voices file. Available: {list(self._voices.keys())}")
            self._voice = next(iter(self._voices.keys()))
            logger.info(f"Falling back to voice: {self._voice}")

        logger.info(f"Kokoro voices loaded. Using voice: {self._voice}")

    def _pick_ref_style(self, tokens_len: int) -> np.ndarray:
        """Select the appropriate style vector based on text length."""
        voice_arr = self._voices[self._voice]
        # Expect shape (N, 1, 256). Clamp index to last row if text is long.
        idx = min(tokens_len, voice_arr.shape[0] - 1)
        ref_s = voice_arr[idx]  # shape (1, 256)
        if ref_s.ndim == 1:
            ref_s = ref_s.reshape(1, -1)
        return ref_s.astype(np.float32)

    def _build_inputs(self, tokens: List[int]) -> Dict[str, np.ndarray]:
        """Build ONNX model inputs from tokenized text."""
        tokens_with_pad = np.array([[0, *tokens, 0]], dtype=np.int64)  # shape (1, <=512)
        ref_s = self._pick_ref_style(len(tokens))
        speed = np.ones(1, dtype=np.float32)

        # Model input names can differ across exports: "input_ids" vs "tokens"
        inputs = {}
        if "input_ids" in self._input_names:
            inputs["input_ids"] = tokens_with_pad
        elif "tokens" in self._input_names:
            inputs["tokens"] = tokens_with_pad
        else:
            # Fallback to first input name
            inputs[self._input_names[0]] = tokens_with_pad

        # Style input name is commonly "style"
        if "style" in self._input_names:
            inputs["style"] = ref_s
        else:
            # Find something that looks like style
            for name in self._input_names:
                if "style" in name:
                    inputs[name] = ref_s
                    break

        # Speed input name typically "speed"
        if "speed" in self._input_names:
            inputs["speed"] = speed
        else:
            for name in self._input_names:
                if "speed" in name:
                    inputs[name] = speed
                    break

        return inputs

    def _synthesize(self, text: str) -> bytes:
        """Synchronously generate audio from text using ONNX model."""
        self._ensure_initialized()

        # Tokenize text to IPA phoneme IDs
        tokens = self._tokenizer(text)
        if len(tokens) > 510:
            # Clamp overly long inputs (model context ~512 incl pads)
            tokens = tokens[:510]
            logger.warning(f"Text too long, truncated to {len(tokens)} tokens")

        inputs = self._build_inputs(tokens)
        outputs = self._sess.run(None, inputs)
        audio = outputs[0]  # shape (1, T) float32 in [-1, 1]
        
        if isinstance(audio, list):
            audio = np.asarray(audio)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        audio = np.clip(audio, -1.0, 1.0)

        # Convert to 16-bit PCM bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Kokoro ONNX.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Run audio generation in executor (separate thread) to avoid blocking
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(None, self._synthesize, text)

            await self.stop_ttfb_metrics()

            # Stream the audio in chunks
            CHUNK_SIZE = self.chunk_size
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i : i + CHUNK_SIZE]
                if len(chunk) > 0:
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    # Small delay to prevent overwhelming the pipeline
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in KokoroOnnxTTSService.run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        # Ensure model is initialized
        self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await super().__aexit__(exc_type, exc_val, exc_tb)
