"""
Ultra Low-Latency Voice Agent with Advanced Performance Optimizations

This implementation focuses on minimizing every possible source of latency:
1. Aggressive VAD settings for instant speech detection
2. Optimized Whisper configuration for speed over accuracy
3. Streaming TTS with minimal chunk sizes
4. Reduced buffer sizes and timeouts
5. Parallel processing where possible
6. Memory and CPU optimizations
"""

import argparse
import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

load_dotenv(override=True)

# Enhanced debug logging configuration
logger.remove()
logger.add(
    sys.stderr, 
    level="DEBUG", 
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    enqueue=True
)
logger.info("🔍 Enhanced debug logging enabled")

# Configure NLTK data path
import nltk
nltk_data_path = os.getenv("NLTK_DATA", "")
if nltk_data_path:
    nltk.data.path.append(nltk_data_path)

# Enable debug logging with performance timing
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from kokoro_tts_onnx import KokoroOnnxTTSService
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

# Import our ultra-optimized streaming TTS
from streaming_tts_v3 import create_ultra_fast_streaming_tts

load_dotenv(override=True)

# Create FastAPI app
app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

# Optimized system instruction for faster processing
SYSTEM_INSTRUCTION = """
You are Pipecat, a fast, helpful AI assistant.

Keep responses VERY brief - 1 short sentence maximum unless asked for more detail. No special characters, markdown, or formatting. Be direct and conversational.

Start by saying "Hi there!" and wait.
"""

# Performance tracking
class PerformanceTracker:
    def __init__(self):
        self.times = {}
    
    def start(self, event: str):
        self.times[event] = time.time()
    
    def end(self, event: str):
        if event in self.times:
            duration = time.time() - self.times[event]
            logger.info(f"⏱️ {event}: {duration*1000:.1f}ms")
            del self.times[event]

perf = PerformanceTracker()

async def run_bot(webrtc_connection):
    perf.start("bot_initialization")
    logger.info("🚀 Starting ultra low-latency bot initialization...")
    logger.debug(f"🔍 [DEBUG] WebRTC connection ID: {webrtc_connection.pc_id}")
    
    # Ultra-aggressive VAD settings for instant detection
    logger.debug("🔍 [DEBUG] Creating SmallWebRTCTransport")
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=0.3,   # Very low for instant detection
                start_secs=0.05,  # Extremely fast start (50ms)
                stop_secs=0.3,    # Quick stop detection
                min_volume=0.3,   # Lower volume threshold
            )),
        ),
    )
    logger.info("⚡ Ultra-aggressive VAD configured")
    logger.debug("🔍 [DEBUG] VAD settings: confidence=0.3, start=50ms, stop=300ms")

    # Ultra-fast Whisper configuration
    logger.info("🎤 Initializing speed-optimized Whisper...")
    logger.debug("🔍 [DEBUG] Using TINY Whisper model for maximum speed")
    stt = WhisperSTTService(
        model=Model.TINY,  # Smallest, fastest model
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        language="en",
        no_speech_threshold=0.4,  # Lower for speed
        initial_prompt="Quick response conversation",
        # Speed optimizations
        condition_on_previous_text=False,  # Disable context for speed
        beam_size=1,  # Single beam for speed
        best_of=1,    # No alternatives for speed
    )
    logger.info("⚡ Ultra-fast Whisper configured")

    # Load and optimize TTS
    logger.info("🔊 Initializing speed-optimized Kokoro TTS...")
    model_path = os.getenv("KOKORO_MODEL_PATH", "")
    voices_path = os.getenv("KOKORO_VOICES_PATH", "")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Kokoro model not found: {model_path}")
        raise FileNotFoundError(f"Kokoro model not found: {model_path}")
        
    if not os.path.exists(voices_path):
        logger.error(f"❌ Kokoro voices not found: {voices_path}")
        raise FileNotFoundError(f"Kokoro voices not found: {voices_path}")
    
    base_tts = KokoroOnnxTTSService(
        model_path=model_path,
        voices_path=voices_path,
        voice="af_heart",
        sample_rate=24000,
    )
    
    # Apply ultra-aggressive streaming
    logger.info("🎵 Applying ultra-fast streaming TTS...")
    tts = create_ultra_fast_streaming_tts(
        base_tts,
        chunk_size=4,        # Micro-chunks (4 words)
        max_delay=0.01       # Ultra-minimal delay (10ms)
    )
    logger.info("⚡ Ultra-streaming TTS ready")

    # Speed-optimized LLM configuration
    logger.info("🧠 Configuring speed-optimized LLM...")
    llm = OpenAILLMService(
        api_key="dummyKey",
        model="gemma3:270m",  # Smallest, fastest model
        base_url="http://127.0.0.1:11434/v1",
        max_tokens=150,       # Reduced for faster generation
        temperature=0.7,      # Balanced for speed and variety
    )
    logger.info("⚡ Speed-optimized LLM ready")

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": SYSTEM_INSTRUCTION,
            }
        ],
    )
    
    # Ultra-fast aggregation settings
    context_aggregator = llm.create_context_aggregator(
        context,
        user_params=LLMUserAggregatorParams(
            aggregation_timeout=0.01,  # Minimal timeout (10ms)
        ),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Optimized pipeline order for minimal latency
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # Performance-optimized task configuration
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,        # Disable metrics for speed
            enable_usage_metrics=False,  # Disable usage tracking
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        perf.start("client_ready_to_bot_ready")
        logger.info("📱 Client ready - activating bot...")
        await rtvi.set_bot_ready()
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        perf.end("client_ready_to_bot_ready")
        logger.info("🎯 Bot activated and ready for ultra-fast responses!")

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        perf.start("participant_join")
        logger.info(f"👤 Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])
        perf.end("participant_join")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"👋 Participant left: {participant}, reason: {reason}")
        await task.cancel()

    perf.end("bot_initialization")
    logger.info("🚀 Ultra low-latency voice agent ready!")
    
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    logger.debug("🔍 [DEBUG] WebRTC /api/offer endpoint called")
    logger.debug(f"🔍 [DEBUG] Request data: {request}")
    perf.start("webrtc_setup")
    logger.info(f"🔗 WebRTC offer received: {request.get('pc_id')}")
    pc_id = request.get("pc_id")
    logger.debug(f"🔍 [DEBUG] Processing PC ID: {pc_id}")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"♻️ Reusing connection: {pc_id}")
        logger.debug("🔍 [DEBUG] Calling renegotiate on existing connection")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        logger.info("🔧 Creating new ultra-fast WebRTC connection...")
        logger.debug("🔍 [DEBUG] Initializing SmallWebRTCConnection")
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        logger.debug("🔍 [DEBUG] Calling initialize on WebRTC connection")
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])
        logger.debug("🔍 [DEBUG] WebRTC connection initialized successfully")

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"🔌 Connection closed: {webrtc_connection.pc_id}")
            logger.debug(f"🔍 [DEBUG] Removing connection {webrtc_connection.pc_id} from pcs_map")
            pcs_map.pop(webrtc_connection.pc_id, None)

        logger.debug("🔍 [DEBUG] Adding background task to run bot pipeline")
        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection
    logger.debug(f"🔍 [DEBUG] Stored connection in pcs_map with PC ID: {answer.get('pc_id')}")
    
    perf.end("webrtc_setup")
    logger.info(f"✅ WebRTC ready: {answer.get('pc_id')}")
    logger.debug(f"🔍 [DEBUG] Returning WebRTC answer: {answer}")
    return answer


@app.get("/api/test-ollama")
async def test_ollama():
    """Test endpoint to check Ollama performance"""
    logger.debug("🔍 [DEBUG] Starting Ollama test endpoint")
    perf.start("ollama_test")
    import httpx
    try:
        logger.debug("🔍 [DEBUG] Creating HTTP client for Ollama")
        async with httpx.AsyncClient() as client:
            logger.debug("🔍 [DEBUG] Sending request to http://127.0.0.1:11434/api/tags")
            response = await client.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
            logger.debug(f"🔍 [DEBUG] Ollama response status: {response.status_code}")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                perf.end("ollama_test")
                logger.info(f"🎯 Ollama ready with {len(model_names)} models")
                logger.debug(f"🔍 [DEBUG] Available models: {model_names}")
                return {"status": "success", "models": model_names}
            else:
                perf.end("ollama_test")
                logger.error(f"❌ [DEBUG] Ollama returned status {response.status_code}")
                logger.debug(f"🔍 [DEBUG] Response content: {response.text}")
                return {"status": "error", "message": f"Status {response.status_code}"}
    except Exception as e:
        perf.end("ollama_test")
        logger.error(f"❌ [DEBUG] Ollama connection error: {str(e)}")
        logger.debug(f"🔍 [DEBUG] Full exception: {type(e).__name__}: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/performance")
async def get_performance_info():
    """Get performance optimization info"""
    return {
        "optimizations": {
            "vad": "Ultra-aggressive (50ms start, 300ms stop)",
            "whisper": "TINY model with speed optimizations",
            "tts": "Streaming with 6-word chunks, 20ms delay",
            "llm": "gemma3:270m with 150 token limit",
            "pipeline": "Metrics disabled for speed",
        },
        "expected_latencies": {
            "speech_detection": "50-100ms",
            "transcription": "100-300ms", 
            "llm_response": "200-800ms",
            "tts_start": "50-150ms",
            "total_perceived": "400-1200ms"
        }
    }


@app.get("/")
async def root():
    """Root endpoint with server info and debug status"""
    logger.debug("🔍 [DEBUG] Root endpoint accessed")
    return {
        "service": "Ultra Low-Latency Voice Agent",
        "version": "v1.0",
        "status": "running",
        "endpoints": {
            "/api/offer": "POST - WebRTC offer endpoint",
            "/api/test-ollama": "GET - Test Ollama connection",
            "/api/performance": "GET - Performance optimization info"
        },
        "debug": {
            "active_connections": len(pcs_map),
            "connection_ids": list(pcs_map.keys()) if pcs_map else []
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Ultra Low-Latency Voice Agent Server")
    yield
    logger.info("🛑 Shutting down voice agent...")
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()



if __name__ == "__main__":
    logger.debug("🔍 [DEBUG] Starting ultra-fast voice agent main execution")
    parser = argparse.ArgumentParser(description="Ultra Low-Latency Voice Agent")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()
    logger.debug(f"🔍 [DEBUG] Parsed arguments - Host: {args.host}, Port: {args.port}")

    logger.info(f"🚀 Starting ultra low-latency voice agent on {args.host}:{args.port}")
    
    # Test Ollama connection before starting server
    logger.debug("🔍 [DEBUG] Testing Ollama connection before server start")
    try:
        import httpx
        import asyncio
        async def test_ollama_startup():
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    logger.info(f"✅ [DEBUG] Ollama accessible. Found {len(models)} models")
                    return True
                else:
                    logger.error(f"❌ [DEBUG] Ollama returned status {response.status_code}")
                    return False
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ollama_ok = loop.run_until_complete(test_ollama_startup())
        loop.close()
        
        if not ollama_ok:
            logger.warning("⚠️ [DEBUG] Ollama not accessible, server may have issues")
    except Exception as e:
        logger.error(f"❌ [DEBUG] Ollama pre-check failed: {e}")
    
    logger.debug("🔍 [DEBUG] Starting uvicorn server")
    uvicorn.run(app, host=args.host, port=args.port)
