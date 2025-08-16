#
# Completely isolated Kokoro TTS using separate process
# Final attempt to avoid Metal threading issues
#

import asyncio
import subprocess
import json
import base64
import tempfile
import os
import signal
import atexit
import sys
from typing import AsyncGenerator, Optional
import threading

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class KokoroTTSIsolated(TTSService):
    """Completely isolated Kokoro TTS using subprocess to avoid Metal issues."""

    _worker_script = None
    _instances = []
    _lock = threading.Lock()

    def __init__(
        self,
        *,
        model: str = "prince-canuma/Kokoro-82M",
        voice: str = "af_heart",
        device: Optional[str] = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        """Initialize the isolated Kokoro TTS service."""
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_name = model
        self._voice = voice
        self._device = device
        
        self._process = None
        self._initialized = False
        
        with KokoroTTSIsolated._lock:
            KokoroTTSIsolated._instances.append(self)
            if KokoroTTSIsolated._worker_script is None:
                KokoroTTSIsolated._create_worker_script()
                atexit.register(KokoroTTSIsolated._cleanup_all)

        self._settings = {
            "model": model,
            "voice": voice,
            "sample_rate": sample_rate,
        }

    @classmethod
    def _create_worker_script(cls):
        """Create the worker script file."""
        worker_code = '''#!/usr/bin/env python3
import sys
import json
import base64
import traceback
import numpy as np

# Add logging to worker
import logging
logging.basicConfig(level=logging.INFO, format='WORKER: %(message)s')

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

class Worker:
    def __init__(self):
        self.model = None
        self.voice = None
        
    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            self.model = load_model(model_name)
            self.voice = voice
            # Test
            list(self.model.generate(text="test", voice=voice, speed=1.0))
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}
    
    def generate(self, text):
        try:
            if not self.model:
                return {"error": "Not initialized"}
            
            segments = []
            for result in self.model.generate(text=text, voice=self.voice, speed=1.0):
                # Convert MLX array to numpy immediately
                audio_data = np.array(result.audio, copy=True)
                print(f"Generated segment shape: {audio_data.shape}, min: {audio_data.min():.4f}, max: {audio_data.max():.4f}", file=sys.stderr)
                segments.append(audio_data)
            
            if not segments:
                return {"error": "No audio"}
                
            # Concatenate all segments
            if len(segments) == 1:
                audio = segments[0]
            else:
                audio = np.concatenate(segments, axis=0)
            
            print(f"Final audio shape: {audio.shape}, min: {audio.min():.4f}, max: {audio.max():.4f}", file=sys.stderr)
            
            # Check if audio is silent
            if np.max(np.abs(audio)) < 1e-6:
                return {"error": "Generated audio is silent"}
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
            
            return {"success": True, "audio": audio_b64}
        except Exception as e:
            import traceback
            return {"error": f"{str(e)}\\n{traceback.format_exc()}"}

worker = Worker()

for line in sys.stdin:
    try:
        req = json.loads(line.strip())
        if req["cmd"] == "init":
            resp = worker.initialize(req["model"], req["voice"])
        elif req["cmd"] == "generate":
            resp = worker.generate(req["text"])
        else:
            resp = {"error": "Unknown command"}
        print(json.dumps(resp), flush=True)
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
'''
        
        # Write to temp file
        fd, cls._worker_script = tempfile.mkstemp(suffix='.py', prefix='kokoro_worker_')
        with os.fdopen(fd, 'w') as f:
            f.write(worker_code)
        os.chmod(cls._worker_script, 0o755)
        logger.info(f"Created worker script: {cls._worker_script}")

    @classmethod
    def _cleanup_all(cls):
        """Clean up all instances."""
        with cls._lock:
            for instance in cls._instances:
                instance._cleanup()
            if cls._worker_script and os.path.exists(cls._worker_script):
                os.unlink(cls._worker_script)

    def _start_worker(self):
        """Start the worker process."""
        try:
            self._process = subprocess.Popen(
                [sys.executable, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            logger.info(f"Started Kokoro worker process: {self._process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            return False

    def _send_command(self, command: dict) -> dict:
        """Send command to worker and get response."""
        try:
            if not self._process or self._process.poll() is not None:
                logger.debug("Starting worker process...")
                if not self._start_worker():
                    return {"error": "Failed to start worker"}

            # Send command
            cmd_json = json.dumps(command) + '\n'
            logger.debug(f"Sending command: {command}")
            self._process.stdin.write(cmd_json)
            self._process.stdin.flush()

            # Read response with timeout
            import select
            ready, _, _ = select.select([self._process.stdout], [], [], 10.0)  # 10 second timeout
            
            if not ready:
                return {"error": "Worker response timeout"}

            response_line = self._process.stdout.readline()
            if not response_line:
                # Check if process died
                if self._process.poll() is not None:
                    stderr_output = self._process.stderr.read() if self._process.stderr else ""
                    return {"error": f"Worker process died. stderr: {stderr_output}"}
                return {"error": "No response from worker"}

            response_data = json.loads(response_line.strip())
            # Don't log the full response if it contains audio data (too verbose)
            if "audio" in response_data:
                logger.debug(f"Worker response: success with {len(response_data.get('audio', ''))} chars of audio data")
            else:
                logger.debug(f"Worker response: {response_line.strip()}")
            return response_data

        except Exception as e:
            logger.error(f"Worker communication error: {e}")
            # Get stderr if available
            if self._process and self._process.stderr:
                try:
                    stderr_output = self._process.stderr.read()
                    logger.error(f"Worker stderr: {stderr_output}")
                except:
                    pass
            return {"error": str(e)}

    async def _initialize_if_needed(self):
        """Initialize the worker if not already done."""
        if self._initialized:
            return True

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._send_command,
            {"cmd": "init", "model": self._model_name, "voice": self._voice}
        )

        if result.get("success"):
            self._initialized = True
            logger.info("Kokoro worker initialized")
            return True
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Worker initialization failed: {error_msg}")
            
            # Also check if process died
            if self._process and self._process.poll() is not None:
                stderr_output = self._process.stderr.read() if self._process.stderr else ""
                logger.error(f"Worker process stderr: {stderr_output}")
            
            return False

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech using isolated worker process."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Initialize worker if needed
            if not await self._initialize_if_needed():
                raise RuntimeError("Failed to initialize Kokoro worker")

            # Generate audio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._send_command,
                {"cmd": "generate", "text": text}
            )

            if not result.get("success"):
                raise RuntimeError(f"Audio generation failed: {result.get('error')}")

            # Decode audio
            audio_b64 = result["audio"]
            audio_bytes = base64.b64decode(audio_b64)

            await self.stop_ttfb_metrics()

            # Stream audio
            CHUNK_SIZE = self.chunk_size
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i : i + CHUNK_SIZE]
                if len(chunk) > 0:
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    def _cleanup(self):
        """Clean up worker process."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except:
                try:
                    self._process.kill()
                except:
                    pass
            self._process = None

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        await self._initialize_if_needed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean shutdown."""
        self._cleanup()
        with KokoroTTSIsolated._lock:
            if self in KokoroTTSIsolated._instances:
                KokoroTTSIsolated._instances.remove(self)
        await super().__aexit__(exc_type, exc_val, exc_tb)