from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from tempfile import TemporaryDirectory
from pydub import AudioSegment

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
async def denoise_audio(file: UploadFile = File(...) ):
    denoiser = AudioDenoiser()

    audio = AudioSegment.from_file_using_temporary_files(file.file)

    with TemporaryDirectory() as temp_dir:
        output_path = f"{temp_dir}/denoised_audio.mp3"

        await denoiser.denoise_audio_file(audio, output_path)

        return FileResponse(output_path)

