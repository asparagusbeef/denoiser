from fastapi import FastAPI, File, UploadFile
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse
from tempfile import TemporaryDirectory, NamedTemporaryFile
from pydub import AudioSegment
from os import unlink

from denoiser import AudioDenoiser

app = FastAPI()

@app.get("/ready")
async def ready():
    return {"message": "Hello World!"}

@app.get("/cuda_is_available")
async def cuda_is_available():
    import torch
    return {"message": torch.cuda.is_available()}

@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(...)):
    denoiser = AudioDenoiser()

    audio = AudioSegment.from_file_using_temporary_files(file.file)

    # Create a temporary file that will not be automatically deleted
    temp_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    output_path = temp_file.name
    temp_file.close()  # Close the file so the denoiser can write to it

    await denoiser.denoise_audio_file(audio, output_path)

    # Define a cleanup function that will delete the temporary file after sending the response
    def cleanup_file(path: str):
        unlink(path)

    # Add the cleanup task to run after sending the response
    background_task = BackgroundTask(cleanup_file, output_path)

    return FileResponse(output_path, filename="denoised_audio.mp3", background=background_task)