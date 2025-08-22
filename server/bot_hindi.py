import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv(override=True)

# Configure NLTK data path
import nltk
nltk_data_path = os.getenv("NLTK_DATA", "")
if nltk_data_path:
    nltk.data.path.append(nltk_data_path)

# Enable debug logging
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

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


load_dotenv(override=True)

app = FastAPI()

# Add CORS middleware to allow client connections from different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


SYSTEM_INSTRUCTION_HINDI = """
आप एक विशेष हिंदी वॉयस असिस्टेंट हैं जो केवल हिंदी में बातचीत करता है।

🇮🇳 आपकी पहचान: आप हिंदी वॉयस बॉट हैं - केवल हिंदी में बात करते हैं
🎤 आप आवाज़ के ज़रिए रियल-टाइम बातचीत करते हैं
🤖 आप एक AI असिस्टेंट हैं जो हिंदी संस्कृति और भाषा को अच्छी तरह समझता है

आपका इनपुट उपयोगकर्ता की आवाज़ से रियल-टाइम में ट्रांस्क्राइब किया गया टेक्स्ट है। ट्रांस्क्रिप्शन की त्रुटियाँ हो सकती हैं। इन त्रुटियों को अपने आप ठीक करके जवाब दें।

आपका आउटपुट ऑडियो में बदला जाएगा इसलिए अपने जवाबों में विशेष वर्ण शामिल न करें और कोई मार्कडाउन या विशेष फॉर्मेटिंग का उपयोग न करें।

महत्वपूर्ण नियम:
- हमेशा केवल हिंदी में जवाब दें, अंग्रेजी का बिल्कुल उपयोग न करें
- यदि कोई अंग्रेजी में बात करे तो कहें "मैं केवल हिंदी में बात करता हूं"
- संक्षिप्त और स्पष्ट उत्तर दें, अधिकतम एक या दो वाक्य में
- विनम्र और सम्मानजनक भाषा का उपयोग करें
- सरल और आसान हिंदी शब्दों का उपयोग करें
- भारतीय संदर्भ में उत्तर दें

बातचीत की शुरुआत "नमस्ते, मैं आपका हिंदी वॉयस असिस्टेंट हूँ!" कहकर करें। फिर रुकें और उपयोगकर्ता का इंतज़ार करें।
"""


async def run_bot(webrtc_connection):
    logger.info("Starting Hindi bot initialization...")
    
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=0.5,  # Lower confidence for better sensitivity  
                start_secs=0.1,  # Faster start detection
                stop_secs=0.5,   # Longer stop time to capture full phrases
                min_volume=0.4,  # Lower volume threshold
            )),
            # Removed smart turn analyzer to avoid dependency issues
        ),
    )
    logger.info("Transport initialized")

    # Hindi Whisper STT (using SMALL model for better Hindi support)
    logger.info("Initializing Whisper STT for Hindi...")
    stt = WhisperSTTService(
        model=Model.SMALL,  # SMALL model has better Hindi support than BASE
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        language="hi",  # Hindi language code
        # Enable debugging
        no_speech_threshold=0.6,  # Lower threshold to be more sensitive
        initial_prompt="नमस्ते, आप कैसे हैं? यह एक हिंदी बातचीत है।",  # Hindi prompt
    )
    logger.info("Hindi Whisper STT initialized")

    # ONNX-based Kokoro TTS (will attempt Hindi pronunciation)
    logger.info("Initializing Kokoro TTS for Hindi...")
    model_path = os.getenv("KOKORO_MODEL_PATH", "")
    voices_path = os.getenv("KOKORO_VOICES_PATH", "")
    
    logger.info(f"Kokoro model path: {model_path}")
    logger.info(f"Kokoro voices path: {voices_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Kokoro model file not found at: {model_path}")
        raise FileNotFoundError(f"Kokoro model file not found: {model_path}")
        
    if not os.path.exists(voices_path):
        logger.error(f"Kokoro voices file not found at: {voices_path}")
        raise FileNotFoundError(f"Kokoro voices file not found: {voices_path}")
    
    tts = KokoroOnnxTTSService(
        model_path=model_path,
        voices_path=voices_path,
        voice="af_heart",  # Using clearest voice for Hindi pronunciation
        sample_rate=24000,
    )
    logger.info("Kokoro TTS initialized for Hindi support")

    logger.info("Initializing LLM service for Hindi...")
    llm = OpenAILLMService(
        api_key="dummyKey",
        model="llama3.2:latest",  # Ollama model name for Llama 3.2 (excellent Hindi support)
        base_url="http://127.0.0.1:11434/v1",  # Ollama default port
        max_tokens=4096,
    )
    logger.info("Hindi LLM service initialized")

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": SYSTEM_INSTRUCTION_HINDI,
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper local service isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Set aggregation_timeout to a
        # a de minimus value since we don't expect any transcript aggregation to be
        # necessary.
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

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

    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    #
    # RTVI event handlers
    #

    def on_client_ready(event):
        logger.info("Hindi client ready event received")
        rtvi.set_client_ready()
        logger.info("Hindi bot marked as ready")
        task.queue_frames([context_aggregator.user().get_context_frame()])
        logger.info("Hindi context frame queued")

    rtvi.add_event_handler("on_client_ready", on_client_ready)

    runner = PipelineRunner()

    await runner.run(task)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/test-hindi")
async def test_hindi():
    """Test endpoint to verify Hindi components"""
    try:
        result = {
            "status": "success",
            "message": "Hindi voice agent components ready",
            "components": {
                "whisper_language": "hi",
                "whisper_model": "small",
                "ollama_model": "llama3.2:latest",
                "tts_voice": "af_heart",
                "system_language": "Hindi"
            },
            "test_phrase": "नमस्ते, मैं आपका हिंदी वॉयस असिस्टेंट हूँ!"
        }
        
        # Test Ollama connection
        import httpx
        async with httpx.AsyncClient() as client:
            ollama_response = await client.get("http://127.0.0.1:11434/api/tags")
            if ollama_response.status_code == 200:
                models = ollama_response.json().get("models", [])
                model_names = [m["name"] for m in models]
                result["ollama_status"] = "connected"
                result["available_models"] = model_names
            else:
                result["ollama_status"] = "disconnected"
                
    except Exception as e:
        result = {
            "status": "error", 
            "message": f"Error testing Hindi components: {str(e)}"
        }
    
    return result


@app.get("/api/test-ollama")
async def test_ollama():
    """Test Ollama connectivity and available models"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"Ollama accessible. Available models: {model_names}")
                return {
                    "status": "success", 
                    "models": model_names,
                    "message": f"Ollama running with {len(model_names)} models"
                }
            else:
                return {"status": "error", "message": f"Ollama returned status {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Cannot connect to Ollama: {str(e)}"}


@app.options("/api/offer")
async def offer_options():
    """Handle CORS preflight requests for /api/offer endpoint"""
    return {"status": "ok"}


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    logger.info(f"Received offer request: pc_id={request.get('pc_id')}")
    
    pc_id = request.get("pc_id")
    
    if pc_id and pc_id in pcs_map:
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        # Reuse existing connection
        pipecat_connection = pcs_map[pc_id]
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        logger.info("Creating new WebRTC connection")
        # Create new connection
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])
        logger.info("WebRTC connection initialized")
        
        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)
        
        # Run bot in background
        logger.info("Starting Hindi bot task in background")
        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    logger.info(f"Generated answer for pc_id: {answer.get('pc_id')}")
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


async def cleanup():
    logger.info("Cleaning up connections...")
    for pc in pcs_map.values():
        await pc.close()
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hindi Voice Agent Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")

    args = parser.parse_args()

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        logger.info("Server shutting down...")
        asyncio.run(cleanup())
