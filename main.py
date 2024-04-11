from fastapi import FastAPI, File, UploadFile, Depends, Request, WebSocket
from fastapi.exceptions import WebSocketException
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from os import environ
from contextlib import asynccontextmanager
from denoiser import AudioDenoiser
from utils import get_gpu_usage
from asyncio import sleep

if environ.get('ENVIRONMENT', 'LOCAL') == 'LOCAL':
    from dotenv import load_dotenv
    load_dotenv()

if environ.get('DEBUG', 'FALSE').upper() == 'TRUE':
    import debugpy # type: ignore

    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()

async def get_denoiser(request: Request) -> AudioDenoiser:
    return request.app.state.denoiser

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.denoiser = AudioDenoiser()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/ready")
async def ready():
    return {"message": "Hello World!"}

@app.get("/cuda_is_available")
async def cuda_is_available():
    import torch
    return {"message": torch.cuda.is_available()}

@app.websocket("/ws/gpu_info")
async def gpu_info(websocket: WebSocket):
    await websocket.accept()
    password = await websocket.receive_text()

    if password != environ['GPU_INFO_PASSWORD']:
        raise WebSocketException(code=1008, detail="Invalid password")

    while True:
        try:
            gpu_usage = get_gpu_usage()
        except ValueError:
            msg = "GPU Access Denied"
            await websocket.send_json({"error": msg})
            sleep(1)
        await websocket.send_json(gpu_usage)
        await sleep(1)

@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(...), denoiser: AudioDenoiser = Depends(get_denoiser)):

    def process_audio():
        audio = AudioSegment.from_file_using_temporary_files(file.file)
        denoised_audio = denoiser.run(audio)
        return denoised_audio

    denoised_audio = process_audio()

    async def iterfile():
        while chunk := denoised_audio.read(1024):
            yield chunk
            await sleep(0.1)

    return StreamingResponse(
        iterfile(),
        media_type="audio/mpeg"
    )