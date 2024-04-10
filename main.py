from fastapi import FastAPI, File, UploadFile, Depends, Request, WebSocket
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from os import unlink, environ
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
        await websocket.send_text("Invalid password")
        return

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
def denoise_audio(file: UploadFile = File(...), denoiser: AudioDenoiser = Depends(get_denoiser)):

    audio = AudioSegment.from_file_using_temporary_files(file.file)

    # Create a temporary file that will not be automatically deleted
    temp_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    output_path = temp_file.name
    temp_file.close()  # Close the file so the denoiser can write to it

    denoiser.denoise_audio_file(audio, output_path)

    # Define a cleanup function that will delete the temporary file after sending the response
    def cleanup_file(path: str):
        unlink(path)

    # Add the cleanup task to run after sending the response
    background_task = BackgroundTask(cleanup_file, output_path)

    return FileResponse(output_path, filename="denoised_audio.mp3", background=background_task)