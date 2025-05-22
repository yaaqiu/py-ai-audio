import os
import requests
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
from test_mix import AudioMixer
import subprocess
import tempfile
import shutil
from pathlib import Path
import imageio_ffmpeg as ffmpeg


def convert_audio_to_wav(input_path, output_path=None):
    """Convert audio file to WAV format using ffmpeg."""
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.wav'))

    try:
        # 使用 ffmpeg 进行转换，添加更多参数以提高兼容性
        subprocess.run([
            'ffmpeg', '-y',  # -y to overwrite output file if it exists
            '-i', input_path,
            '-acodec', 'pcm_s16le',  # Use PCM 16-bit encoding
            '-ar', '44100',  # Set sample rate to 44.1kHz
            '-ac', '1',  # Convert to mono
            '-vn',  # Disable video
            '-f', 'wav',  # Force WAV format
            output_path
        ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e.stderr.decode()}")
        return None


def trim_audio(input_path, duration, output_path=None):
    """Trim audio file to specified duration using ffmpeg."""
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.trimmed.wav'))

    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', input_path,
            '-t', str(duration),  # Duration in seconds
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '1',
            output_path
        ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error trimming audio1: {e.stderr.decode()}")
        return None


def mix_and_generate(audio_files, output_path, duration=30):
    """Mix multiple audio files and generate a new track."""
    if not audio_files:
        print("No tracks to mix.")
        return None

    # Convert all input files to WAV format
    wav_files = []
    for audio_file in audio_files:
        wav_path = convert_audio_to_wav(audio_file)
        if wav_path:
            wav_files.append(wav_path)

    if not wav_files:
        print("Failed to convert any audio files.")
        return None

    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Trim all files to the specified duration
        trimmed_files = []
        for wav_file in wav_files:
            trimmed_path = os.path.join(temp_dir, f"trimmed_{os.path.basename(wav_file)}")
            trimmed_file = trim_audio(wav_file, duration, trimmed_path)
            if trimmed_file:
                trimmed_files.append(trimmed_file)

        if not trimmed_files:
            print("Failed to trim any audio files.")
            return None

        # Mix the trimmed files
        try:
            # Create a complex ffmpeg filter for mixing
            filter_complex = ' '.join([f'[{i}:a]' for i in range(len(trimmed_files))])
            filter_complex += f' amix=inputs={len(trimmed_files)}:duration=longest:normalize=0'

            subprocess.run([
                'ffmpeg', '-y',
                *[item for pair in zip(['-i'] * len(trimmed_files), trimmed_files) for item in pair],
                '-filter_complex', filter_complex,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                '-f', 'wav',
                output_path
            ], check=True, capture_output=True)

            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error mixing audio: {e.stderr.decode()}")
            return None
        finally:
            # Clean up temporary WAV files
            for wav_file in wav_files:
                try:
                    os.remove(wav_file)
                except OSError:
                    pass


class StabilityAudioGenerator:
    def __init__(self):
        self.api_key = "sk-TTeDHIeFdf3rUGAIGWJO5qKhGOmA8CC4o2bhuQiMXHmaYHh7"

    def trim_audio(self, input_audio, output_audio, duration=180):
        """裁剪音频到指定长度"""
        try:
            # 使用 ffmpeg 进行裁剪
            ffmpeg_path = ffmpeg.get_ffmpeg_exe()
            command = [
                # 'ffmpeg',
                ffmpeg_path,
                '-i', input_audio,
                '-t', str(duration),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                '-y',
                output_audio
            ]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"FFmpeg trimming failed: {stderr.decode()}")
                return False

            return True
        except Exception as e:
            print(f"Error trimming audio2: {str(e)}")
            return False

    def generate_audio(self, input_audio, output_path, prompt="Ambient sleep music, peaceful and calming",
                       duration=180):
        """使用 Stability AI 生成音频"""
        try:
            # 首先裁剪输入音频
            temp_audio = input_audio + '.temp.wav'
            if not self.trim_audio(input_audio, temp_audio, duration=min(180, duration)):
                print("Failed to trim audio")
                return False

            response = requests.post(
                "https://api.stability.ai/v2beta/audio/stable-audio-2/audio-to-audio",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "audio/*"
                },
                files={"audio": open(temp_audio, "rb")},
                data={
                    "prompt": prompt,
                    "duration": min(180, duration),  # 确保不超过180秒
                    "seed": 0,
                    "steps": 50,
                    "cfg_scale": 7.0,
                    "output_format": "mp3",
                    "strength": 0.7
                }
            )

            # 清理临时文件
            try:
                os.remove(temp_audio)
            except:
                pass

            if not response.ok:
                print(f"Generation failed: {response.status_code} - {response.text}")
                return False

            # 保存生成的音频
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Successfully saved audio to: {output_path}")
            return True

        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return False


def mix_and_generate():
    # 创建音频混合器
    mixer = AudioMixer()

    # 添加音轨
    tracks = [
        {
            "path": r"E:\python\ai_audio\audio\喜马拉雅雨声.wav",
            "name": "Rain",
            "volume": 0.25
        },
        {
            "path": r"E:\python\ai_audio\audio\喜马拉雅风声.wav",
            "name": "Wind",
            "volume": 0.3
        }
    ]

    # 添加所有音轨
    for track in tracks:
        if os.path.exists(track["path"]):
            print(f"Processing track: {track['name']}")
            # Convert audio to WAV
            wav_path = convert_audio_to_wav(track["path"])
            if wav_path:
                if mixer.add_track(wav_path, track["name"], track["volume"]):
                    print(f"Successfully added track: {track['name']}")
                else:
                    print(f"Failed to add track: {track['name']}")
            else:
                print(f"Failed to convert track: {track['name']}")
        else:
            print(f"Track file not found: {track['path']}")
            return

    # 生成混合音频
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mixed_file = os.path.join('mixed', f'mixed_{timestamp}.wav')

    if mixer.generate_mix(mixed_file, mix_style='natural'):
        print(f"Successfully generated mix")
        print(f"Mixed file saved to: {mixed_file}")

        # 使用 Stability AI 生成新的音频版本
        generator = StabilityAudioGenerator()

        # 生成不同风格的版本
        versions = [
            {
                "name": "ambient",
                "prompt": "Ambient sleep music with nature sounds, very peaceful and calming, incorporating rain and wind sounds",
                "duration": 180
            },
            {
                "name": "lofi",
                "prompt": "Lofi hip hop beat with rain and wind ambience, relaxing and meditative",
                "duration": 180
            },
            {
                "name": "meditation",
                "prompt": "Deep meditation music with rain sounds, peaceful drone with natural elements",
                "duration": 180
            }
        ]

        for version in versions:
            output_file = os.path.join('mixed', f'stability_{version["name"]}_{timestamp}.mp3')
            print(f"\nGenerating {version['name']} version...")

            if generator.generate_audio(
                    mixed_file,
                    output_file,
                    prompt=version["prompt"],
                    duration=version["duration"]
            ):
                print(f"Successfully generated {version['name']} version")
            else:
                print(f"Failed to generate {version['name']} version")

    else:
        print("Failed to generate mix")


if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs('mixed', exist_ok=True)

    # 运行混音和生成
    mix_and_generate() 