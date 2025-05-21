import os
import requests
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
from test_mix import AudioMixer

class StabilityAudioGenerator:
    def __init__(self):
        self.api_key = "sk-cISiSKqYBMP4WMFh2fQfQYHpm4LVGD7SZFY0HOO3wvSQ9J2C"
        
    def trim_audio(self, input_audio, output_audio, duration=180):
        """裁剪音频到指定长度"""
        try:
            # 加载音频
            y, sr = librosa.load(input_audio, sr=None)
            
            # 计算要保留的样本数
            samples_to_keep = int(duration * sr)
            
            # 如果音频太长，只保留前面的部分
            if len(y) > samples_to_keep:
                y = y[:samples_to_keep]
            
            # 保存裁剪后的音频
            sf.write(output_audio, y, sr)
            return True
        except Exception as e:
            print(f"Error trimming audio: {str(e)}")
            return False
        
    def generate_audio(self, input_audio, output_path, prompt="Ambient sleep music, peaceful and calming", duration=180):
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
            "path": r"C:\Users\dorot\Desktop\suno\audio\喜马拉雅雨声.mp3", 
            "name": "Rain", 
            "volume": 0.25
        },
        {
            "path": r"C:\Users\dorot\Desktop\suno\audio\喜马拉雅风声.mp3", 
            "name": "Wind", 
            "volume": 0.3
        }
    ]
    
    # 添加所有音轨
    for track in tracks:
        if os.path.exists(track["path"]):
            print(f"Adding track: {track['name']}")
            mixer.add_track(track["path"], track["name"], track["volume"])
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