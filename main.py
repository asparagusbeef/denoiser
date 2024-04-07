from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from tempfile import TemporaryDirectory
from pydub import AudioSegment

from denoiser import AudioDenoiser

app = FastAPI()

@app.post("/denoise", response_model=FileResponse)
async def denoise_audio(file: UploadFile = File(...) ):
    denoiser = AudioDenoiser()

    audio = AudioSegment.from_file_using_temporary_files(file.file)

    with TemporaryDirectory() as temp_dir:
        output_path = f"{temp_dir}/denoised_audio.mp3"

        await denoiser.denoise_audio_file(audio, output_path)

        return FileResponse(output_path)

