import librosa
import soundfile as sf
import numpy as np
import os
from datetime import datetime
import librosa
import soundfile as sf
import numpy as np
import os
from datetime import datetime
import json

class AudioMixer:
    def __init__(self):
        self.audio_tracks = []
        self.mix_settings = {
            'sample_rate': 44100,
            'bitrate': 256000,
            'format': 'wav'
        }
    
    def add_track(self, file_path, track_name, volume=1.0, position=0):
        """添加音轨到混合列表"""
        try:
            # 加载音频文件
            audio, sr = librosa.load(file_path, sr=self.mix_settings['sample_rate'])
            
            # 分析音频特征
            features = self.analyze_audio_features(audio, sr)
            
            track_info = {
                'name': track_name,
                'path': file_path,
                'audio': audio,
                'sample_rate': sr,
                'volume': volume,
                'position': position,
                'features': features
            }
            
            self.audio_tracks.append(track_info)
            print(f"Added track: {track_name}")
            return True
        except Exception as e:
            print(f"Error adding track {track_name}: {str(e)}")
            return False
    
    def analyze_audio_features(self, audio, sr):
        """分析音频特征"""
        # 计算频谱
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # 计算频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # 计算频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # 计算过零率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_bandwidth': np.mean(spectral_bandwidth),
            'zero_crossing_rate': np.mean(zero_crossing_rate),
            'spectrum': S_db
        }
    
    def generate_mix(self, output_path, mix_style='natural'):
        """生成混合音频"""
        try:
            if not self.audio_tracks:
                print("No tracks to mix")
                return False
            
            print("\nStarting audio mix...")
            print(f"Mix style: {mix_style}")
            
            # 找到最长的音频长度
            max_length = max(len(track['audio']) for track in self.audio_tracks)
            
            # 创建混合音频数组
            mixed_audio = np.zeros(max_length)
            
            # 根据混合风格处理每个音轨
            for track in self.audio_tracks:
                print(f"\nProcessing track: {track['name']}")
                
                # 获取音频数据
                audio = track['audio']
                
                # 确保长度一致
                if len(audio) < max_length:
                    audio = np.pad(audio, (0, max_length - len(audio)))
                else:
                    audio = audio[:max_length]
                
                # 应用音量
                audio = audio * track['volume']
                
                # 根据混合风格处理音频
                if mix_style == 'natural':
                    # 自然混合：保持原始特性
                    pass
                elif mix_style == 'ambient':
                    # 环境音效：添加混响效果
                    audio = self.add_reverb(audio, track['sample_rate'])
                elif mix_style == 'dynamic':
                    # 动态混合：根据频谱特征调整
                    audio = self.dynamic_mix(audio, track['features'])
                
                # 叠加到混合音频
                mixed_audio += audio
            
            # 归一化防止爆音
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
            
            # 保存混合后的音频
            sf.write(output_path, mixed_audio, self.mix_settings['sample_rate'])
            print(f"\nSuccessfully generated mix")
            print(f"Output saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating mix: {str(e)}")
            return False
    
    def add_reverb(self, audio, sr):
        """添加混响效果"""
        # 简单的混响实现
        reverb_time = 1.0  # 混响时间（秒）
        reverb_samples = int(reverb_time * sr)
        reverb = np.zeros(reverb_samples)
        
        # 创建混响衰减
        decay = np.exp(-np.linspace(0, 5, reverb_samples))
        reverb = decay * np.random.randn(reverb_samples)
        
        # 应用混响
        audio_with_reverb = np.convolve(audio, reverb, mode='full')[:len(audio)]
        return audio_with_reverb
    
    def dynamic_mix(self, audio, features):
        """动态混合处理"""
        # 基于频谱特征调整音频
        spectral_centroid = features['spectral_centroid']
        spectral_bandwidth = features['spectral_bandwidth']
        
        # 计算动态范围压缩
        compression_ratio = 1.0 / (1.0 + spectral_bandwidth / 1000)
        audio = np.sign(audio) * np.power(np.abs(audio), compression_ratio)
        
        return audio

def main():
    # 创建音频混合器
    mixer = AudioMixer()
    
    # 添加音轨
    tracks = [
        {"path": r"D:\2\风吹树叶.mp3", "name": "Wind", "volume": 0.7},
        {"path": r"D:\2\鸟叫.mp3", "name": "Birds", "volume": 0.5},
        {"path": r"D:\2\雨声.mp3", "name": "Rain", "volume": 0.6}
    ]
    
    # 添加所有音轨
    for track in tracks:
        if os.path.exists(track["path"]):
            mixer.add_track(track["path"], track["name"], track["volume"])
        else:
            print(f"Track file not found: {track['path']}")
            return
    
    # 生成混合音频
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join('mixed', f'mixed_{timestamp}.wav')
    
    # 选择混合风格
    mix_style = 'natural'  # 可选: 'natural', 'ambient', 'dynamic'
    
    if mixer.generate_mix(output_file, mix_style):
        print("\nAudio mixing completed successfully!")
        print(f"Output file: {output_file}")
    else:
        print("\nFailed to mix audio files")

if __name__ == '__main__':
    main()
def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV using librosa"""
    try:
        # 加载MP3文件
        audio, sr = librosa.load(mp3_path, sr=None)
        # 保存为WAV
        sf.write(wav_path, audio, sr)
        print(f"Successfully converted {mp3_path} to {wav_path}")
        return True
    except Exception as e:
        print(f"Error converting {mp3_path}: {str(e)}")
        return False

def analyze_audio_volume(audio):
    """分析音频的音量特征"""
    # 计算RMS能量
    rms = librosa.feature.rms(y=audio)[0]
    # 计算平均能量
    avg_rms = np.mean(rms)
    # 计算峰值
    peak = np.max(np.abs(audio))
    return {
        'rms': avg_rms,
        'peak': peak,
        'dynamic_range': peak / avg_rms if avg_rms > 0 else 0
    }

def normalize_audio(audio, target_rms=0.1):
    """将音频归一化到目标RMS值"""
    current_rms = np.sqrt(np.mean(audio**2))
    if current_rms > 0:
        return audio * (target_rms / current_rms)
    return audio

def mix_audio_files(file1, file2, output_path, ratio1=0.5, ratio2=0.5):
    """Mix two audio files with specified ratios and volume balancing"""
    try:
        # 加载两个音频文件
        audio1, sr1 = librosa.load(file1, sr=None)
        audio2, sr2 = librosa.load(file2, sr=None)
        
        print(f"File 1: {os.path.basename(file1)}")
        print(f"Sample rate: {sr1}, Length: {len(audio1)}")
        print(f"File 2: {os.path.basename(file2)}")
        print(f"Sample rate: {sr2}, Length: {len(audio2)}")
        
        # 分析原始音量
        vol1 = analyze_audio_volume(audio1)
        vol2 = analyze_audio_volume(audio2)
        print(f"\nOriginal volumes:")
        print(f"File 1 - RMS: {vol1['rms']:.4f}, Peak: {vol1['peak']:.4f}")
        print(f"File 2 - RMS: {vol2['rms']:.4f}, Peak: {vol2['peak']:.4f}")
        
        # 确保采样率一致
        sr = min(sr1, sr2)
        if sr1 != sr2:
            print(f"Resampling to {sr} Hz...")
            audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=sr)
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr)
        
        # 确保长度一致
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # 音量平衡处理
        # 1. 首先将两个音频归一化到相同的RMS值
        target_rms = 0.1  # 目标RMS值
        audio1 = normalize_audio(audio1, target_rms)
        audio2 = normalize_audio(audio2, target_rms)
        
        # 2. 应用混合比例
        mixed = ratio1 * audio1 + ratio2 * audio2
        
        # 3. 最终归一化防止爆音
        mixed = mixed / np.max(np.abs(mixed))
        
        # 分析混合后的音量
        mixed_vol = analyze_audio_volume(mixed)
        print(f"\nMixed audio volume:")
        print(f"RMS: {mixed_vol['rms']:.4f}, Peak: {mixed_vol['peak']:.4f}")
        
        # 保存混合后的音频
        sf.write(output_path, mixed, sr)
        print(f"\nSuccessfully mixed audio files")
        print(f"Output saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error mixing audio: {str(e)}")
        return False

def main():
    # 创建必要的目录
    os.makedirs('audio', exist_ok=True)
    os.makedirs('mixed', exist_ok=True)
    
    # 指定要混合的音频文件
    mp3_files = [
        r"D:\2\鸟叫.mp3",  # 雨声
        r"D:\2\风吹树叶.mp3"       # 风声
    ]
    
    print("Files to mix:")
    for file in mp3_files:
        print(f"- {file}")
    
    # 转换MP3到WAV
    wav_files = []
    for mp3_path in mp3_files:
        wav_path = os.path.join('audio', os.path.basename(mp3_path).replace('.mp3', '.wav'))
        if not os.path.exists(wav_path):
            if convert_mp3_to_wav(mp3_path, wav_path):
                wav_files.append(wav_path)
        else:
            wav_files.append(wav_path)
    
    # 混合音频
    if len(wav_files) == 2:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join('mixed', f'mixed_{timestamp}.wav')
        
        # 可以调整混合比例
        rain_ratio = 0.5  # 雨声比例
        wind_ratio = 0.5  # 风声比例
        
        if mix_audio_files(wav_files[0], wav_files[1], output_file, rain_ratio, wind_ratio):
            print("\nAudio mixing completed successfully!")
            print(f"Output file: {output_file}")
        else:
            print("\nFailed to mix audio files")
    else:
        print("Error: Need exactly 2 audio files for mixing")

if __name__ == '__main__':
    main() 