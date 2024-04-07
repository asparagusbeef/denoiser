from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment
from tempfile import TemporaryDirectory
from os import path, makedirs

class AudioDenoiser:
    def __init__(self):
        self.model, self.df_state, _ = init_df()

    async def split_to_chunks(self, audio: AudioSegment, chunk_seconds: int = 120, output_dir: str = "chunks") -> list[str]:
        print("Splitting audio to chunks")

        if not path.exists(output_dir):
            makedirs(output_dir)

        chunk_duration_ms = chunk_seconds * 1000

        file_paths = []
        for split_number, i in enumerate(range(0, len(audio), chunk_duration_ms)):
            
            chunk = audio[i : i + chunk_duration_ms]

            chunk.export(f"{output_dir}/chunk_{split_number}.mp3", format="mp3")
            file_paths.append(f"{output_dir}/chunk_{split_number}.mp3")

        return file_paths
      
    async def denoise(self, file_paths: list[str]):
        print("Denoising audio file")

        file_paths_wav = [file_path.rsplit(".", 1)[0] + ".wav" for file_path in file_paths]

        for i, (file_path, file_path_wav) in enumerate(zip(file_paths, file_paths_wav)):

            file, _ = load_audio(file_path, sr=self.df_state.sr())

            denoised_audio = enhance(self.model, self.df_state, file)
            
            save_audio(file_path_wav, denoised_audio, sr=self.df_state.sr())

        return file_paths_wav

    async def reattach_chunks(self, file_paths: list[str]):
        print("Reattaching audio chunks")
        for i, file_path in enumerate(file_paths):
            if i == 0:
                audio = AudioSegment.from_file_using_temporary_files(file_path)
            else:
                audio += AudioSegment.from_file_using_temporary_files(file_path)

        return audio
    
    async def denoise_audio_file(self, audio: AudioSegment, output_path: str = "denoised_audio.mp3"):

        output_dir = output_path.rsplit("/", 1)[0]
        if not path.exists(output_dir):
            makedirs(output_dir)

        print("Denoising audio file")
        denoised_files = await self.denoise(await self.split_to_chunks(audio))

        print("Audio file denoised")
        print("Reattaching audio chunks")
        denoised_audio = await self.reattach_chunks(denoised_files)

        denoised_audio.export(output_path, format="mp3")