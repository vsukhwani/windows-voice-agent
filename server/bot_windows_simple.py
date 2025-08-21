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

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


SYSTEM_INSTRUCTION = """
"You are Pipecat, a friendly, helpful chatbot.

Your input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.

Your output will be converted to audio so don't include special characters in your answers and do not use any markdown or special formatting.

Respond to what the user said in a creative and helpful way. Keep your responses brief unless you are explicitly asked for long or detailed responses. Normally you should use one or two sentences at most. Keep each sentence short. Prefer simple sentences. Try not to use long sentences with multiple comma clauses.

Start the conversation by saying, "Hello, I'm Pipecat!" Then stop and wait for the user.
"""


async def run_bot(webrtc_connection):
    logger.info("Starting bot initialization...")
    
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

    # Local Whisper (faster-whisper backend) on Windows
    logger.info("Initializing Whisper STT...")
    stt = WhisperSTTService(
        model=Model.BASE,  # Using smaller, faster model for better performance
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        language="en",  # Explicitly set language
        # Enable debugging
        no_speech_threshold=0.6,  # Lower threshold to be more sensitive
        initial_prompt="Hello, how are you today? This is a conversation.",  # Help with transcription
    )
    logger.info("Whisper STT initialized")

    # ONNX-based Kokoro TTS (cross-platform)
    logger.info("Initializing Kokoro TTS...")
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
        voice="af_heart",
        sample_rate=24000,
    )
    logger.info("Kokoro TTS initialized successfully")

    logger.info("Initializing LLM service...")
    llm = OpenAILLMService(
        api_key="dummyKey",
        model="gemma3:12b",  # Ollama model name for Gemma 3 12B
        base_url="http://127.0.0.1:11434/v1",  # Ollama default port
        max_tokens=4096,
    )
    logger.info("LLM service initialized")

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": SYSTEM_INSTRUCTION,
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
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Client ready event received")
        await rtvi.set_bot_ready()
        logger.info("Bot marked as ready")
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        logger.info("Context frame queued")

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])
        logger.info("Participant transcription capture started")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()
        logger.info("Task cancelled")

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    logger.info(f"Received offer request: pc_id={request.get('pc_id')}")
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        logger.info("Creating new WebRTC connection")
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])
        logger.info("WebRTC connection initialized")

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        logger.info("Starting bot task in background")
        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    logger.info(f"Generated answer for pc_id: {answer.get('pc_id')}")
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@app.get("/api/test-ollama")
async def test_ollama():
    """Test endpoint to check if Ollama is accessible"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:11434/api/tags", timeout=10.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                logger.info(f"Ollama accessible. Available models: {model_names}")
                return {"status": "success", "models": model_names}
            else:
                logger.error(f"Ollama returned status code: {response.status_code}")
                return {"status": "error", "message": f"Ollama returned status {response.status_code}"}
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return {"status": "error", "message": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner (Windows Simple)")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
