from df.enhance import enhance, init_df, save_audio, resample, ModelParams, AudioDataset, DataLoader
from pydub import AudioSegment
from pydub.utils import make_chunks
from os import path, makedirs, environ
from io import BytesIO
from tempfile import TemporaryDirectory
import time

class AudioDenoiser:
    def __init__(self):
        self.model, self.df_state, _ = init_df(model_base_dir="model")

    def split_to_chunks(self, audio: AudioSegment, output_dir: str, chunk_seconds: int) -> list[str]:
        print("Splitting audio to chunks")
        chunk_duration_ms = chunk_seconds * 1000
        chunks = make_chunks(audio, chunk_duration_ms)

        makedirs(output_dir, exist_ok=True)

        chunk_names = []

        for i, chunk in enumerate(chunks):
            chunk_name = f"chunk_{i}.wav"
            chunk_file_path = path.join(output_dir, chunk_name)
            chunk.export(chunk_file_path, format="wav")
            chunk_names.append(chunk_file_path)

        return chunk_names
      
    def denoise(self, file_paths: list[str], num_workers: int = 2) -> None:

        base_dir = path.dirname(file_paths[0])

        df_sr = ModelParams().sr
        ds = AudioDataset(file_paths, df_sr)
        loader = DataLoader(ds, num_workers=num_workers, pin_memory=True)
        n_samples = len(ds)
        for i, (file, audio, audio_sr) in enumerate(loader):
            file = file[0]
            audio = audio.squeeze(0)
            progress = (i + 1) / n_samples * 100
            t0 = time.time()
            audio = enhance(self.model, self.df_state, audio)
            t1 = time.time()
            t_audio = audio.shape[-1] / df_sr
            t = t1 - t0
            rtf = t / t_audio
            fn = path.basename(file)
            p_str = f"{progress:2.0f}% | " if n_samples > 1 else ""
            print(f"{p_str}Enhanced noisy audio file '{fn}' in {t:.2f}s (RT factor: {rtf:.3f})")
            audio = resample(audio.to("cpu"), df_sr, audio_sr)
            save_audio(file, audio, sr=audio_sr, output_dir=base_dir)

    def reattach_chunks(self, file_paths: list[str]) -> AudioSegment:
        print("Reattaching audio chunks")
        audio = AudioSegment.empty()
        for i, file_path in enumerate(file_paths):
            print(f"Reattaching chunk {i + 1}/{len(file_paths)}")
            audio += AudioSegment.from_file_using_temporary_files(file_path)

        return audio
    
    def run(self, audio: AudioSegment) -> BytesIO:

        NUM_WORKERS = 2

        print("Denoising audio file")
        with TemporaryDirectory() as temp_dir:
            print("Splitting audio to chunks")
            chunk_names = self.split_to_chunks(audio, temp_dir, min(environ.get("CHUNK_SECONDS", 200), len(audio) // NUM_WORKERS // 1000 + 1))

            file_paths = [path.join(temp_dir, chunk_name) for chunk_name in chunk_names]

            print("Denoising audio chunks")
            self.denoise(file_paths, num_workers=NUM_WORKERS)

            print("Audio file denoised")
            print("Reattaching audio chunks")

            denoised_audio = self.reattach_chunks(file_paths)
            print("Audio chunks reattached")

        denoised_audio_buffered = BytesIO(denoised_audio.export(format="mp3").read())
        denoised_audio_buffered.seek(0)
        return denoised_audio_buffered