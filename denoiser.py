from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment
from pydub.utils import make_chunks
from os import path, makedirs
from io import BytesIO
from tempfile import TemporaryDirectory

class AudioDenoiser:
    def __init__(self):
        self.model, self.df_state, _ = init_df(model_base_dir="model")

    def split_to_chunks(self, audio: AudioSegment, output_dir: str, chunk_seconds: int = 120) -> list[str]:
        print("Splitting audio to chunks")
        chunk_duration_ms = chunk_seconds * 1000
        chunks = make_chunks(audio, chunk_duration_ms)

        makedirs(output_dir, exist_ok=True)

        file_paths = []
        for i, chunk in enumerate(chunks):
            chunk_file_path = path.join(output_dir, f"chunk_{i}.mp3")
            chunk.export(chunk_file_path, format="mp3")
            file_paths.append(chunk_file_path)

        return file_paths
      
    def denoise(self, file_paths: list[str]):
        print("Denoising audio file")

        file_paths_wav = [file_path.rsplit(".", 1)[0] + ".wav" for file_path in file_paths]

        for file_path, file_path_wav in zip(file_paths, file_paths_wav):

            file, _ = load_audio(file_path, sr=self.df_state.sr())
            denoised_audio = enhance(self.model, self.df_state, file)
            save_audio(file_path_wav, denoised_audio, sr=self.df_state.sr())

        return file_paths_wav

    def reattach_chunks(self, file_paths: list[str]) -> AudioSegment:
        print("Reattaching audio chunks")
        audio = AudioSegment.empty()
        for i, file_path in enumerate(file_paths):
            print(f"Reattaching chunk {i + 1}/{len(file_paths)}")
            audio += AudioSegment.from_file_using_temporary_files(file_path)

        return audio
    
    def run(self, audio: AudioSegment) -> BytesIO:

        print("Denoising audio file")
        with TemporaryDirectory() as temp_dir:
            print("Splitting audio to chunks")
            audio_chunks = self.split_to_chunks(audio, temp_dir)

            print("Denoising audio chunks")
            denoised_files = self.denoise(audio_chunks)

            print("Audio file denoised")
            print("Reattaching audio chunks")

            denoised_audio = self.reattach_chunks(denoised_files)

        denoised_audio_buffered = BytesIO(denoised_audio.export(format="mp3").read())
        denoised_audio_buffered.seek(0)
        return denoised_audio_buffered