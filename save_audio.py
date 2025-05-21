import requests
import base64
import json
import os
import time
import subprocess
from pathlib import Path
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import librosa
import numpy as np
import soundfile as sf
from datetime import datetime
from openai import OpenAI

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_session():
    """创建一个带有重试机制的会话"""
    session = requests.Session()
    
    # 配置重试策略
    retry_strategy = Retry(
        total=5,  # 最大重试次数
        backoff_factor=1,  # 重试间隔
        status_forcelist=[429, 500, 502, 503, 504]  # 需要重试的HTTP状态码
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

def upload_audio(file_path, purpose='song'):
    """
    上传音频文件并获取voice_id和instrumental_id
    
    Args:
        file_path (str): 音频文件路径
        purpose (str): 文件用途，可选值：song/voice/instrumental
    
    Returns:
        dict: 包含voice_id和instrumental_id的字典，如果失败返回None
    """
    url = "https://api.minimax.chat/v1/music_upload"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLkvp3nhLYiLCJVc2VyTmFtZSI6IuS-neeEtiIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTEwMzE2MTE3OTQzOTE5NTgwIiwiUGhvbmUiOiIxODA2MTYxMzkzMSIsIkdyb3VwSUQiOiIxOTEwMzE2MTE3OTM1NTMwOTcyIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDQtMTIgMTE6Mjk6NDQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.PdNCgvEmgMS6wdjSNKe4MmQKtSlYMkWfFA9pSGrysU47X7UPKbAD-jwBOdHLv9kKuP8VsTOsSTS8al-rNc-M2xYx4xOiIl4rEvO6IXp9-XIEqbr6DtwDNSFoY6SRwLAMZWMQnLpAWXVqwFu6PkB4Wb9nPX3ehLjgVFk1nRQRXwMha9QozptsLrFllrI1ZnJGiBbZ8xcCnbRqNmfE_-XDGK75XyiFWyP_XzzHX_SUn812LEoh7eD_WDoi-anOEfTmzQYlmqGw0SgxTO08LGWKOw2IvHVB4xWrd6Y7SbrJLxzQ4X_22jJvXSnmPB5OHNj-4s5uBxEaarmH9dij65NfCg"
    }

    try:
        # 验证文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        # 验证文件格式
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.mp3', '.wav']:
            print(f"不支持的文件格式: {file_ext}")
            return None

        # 验证文件大小和时长
        file_size = os.path.getsize(file_path)
        if file_size < 1024:  # 文件太小，可能无效
            print("文件太小，可能无效")
            return None

        # 准备上传文件
        files = {
            'file': ('audio' + file_ext, open(file_path, 'rb'), 'audio/' + file_ext[1:])
        }
        
        data = {
            'purpose': purpose
        }

        print(f"正在上传文件: {file_path}")
        print(f"文件大小: {file_size} 字节")
        print(f"上传用途: {purpose}")

        # 创建会话并发送请求
        session = create_session()
        response = session.post(
            url, 
            headers=headers, 
            data=data, 
            files=files,
            verify=False,  # 禁用 SSL 验证
            timeout=30  # 设置超时时间
        )
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        
        # 检查API返回状态
        if "base_resp" not in result or result["base_resp"].get("status_code") != 0:
            print(f"API错误: {result.get('base_resp', {}).get('status_msg', '未知错误')}")
            return None

        # 获取返回的ID
        response_data = {}
        if purpose in ['song', 'voice'] and 'voice_id' in result:
            response_data['voice_id'] = result['voice_id']
        if purpose in ['song', 'instrumental'] and 'instrumental_id' in result:
            response_data['instrumental_id'] = result['instrumental_id']

        print("上传成功！")
        for key, value in response_data.items():
            print(f"{key}: {value}")
        
        return response_data

    except requests.exceptions.SSLError as e:
        print(f"SSL连接错误: {e}")
        print("尝试重新连接...")
        try:
            # 重试一次，使用不同的 SSL 配置
            session = requests.Session()
            session.verify = False
            response = session.post(
                url,
                headers=headers,
                data=data,
                files=files,
                timeout=30
            )
            # ... 处理响应 ...
            result = response.json()
            if "base_resp" in result and result["base_resp"].get("status_code") == 0:
                response_data = {}
                if purpose in ['song', 'voice'] and 'voice_id' in result:
                    response_data['voice_id'] = result['voice_id']
                if purpose in ['song', 'instrumental'] and 'instrumental_id' in result:
                    response_data['instrumental_id'] = result['instrumental_id']
                return response_data
        except Exception as retry_e:
            print(f"重试失败: {retry_e}")
        return None
    except Exception as e:
        print(f"上传文件时出错: {e}")
        return None

def generate_music():
    url = "https://api.minimax.chat/v1/music_generation"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLkvp3nhLYiLCJVc2VyTmFtZSI6IuS-neeEtiIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTEwMzE2MTE3OTQzOTE5NTgwIiwiUGhvbmUiOiIxODA2MTYxMzkzMSIsIkdyb3VwSUQiOiIxOTEwMzE2MTE3OTM1NTMwOTcyIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDQtMTIgMTE6Mjk6NDQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.PdNCgvEmgMS6wdjSNKe4MmQKtSlYMkWfFA9pSGrysU47X7UPKbAD-jwBOdHLv9kKuP8VsTOsSTS8al-rNc-M2xYx4xOiIl4rEvO6IXp9-XIEqbr6DtwDNSFoY6SRwLAMZWMQnLpAWXVqwFu6PkB4Wb9nPX3ehLjgVFk1nRQRXwMha9QozptsLrFllrI1ZnJGiBbZ8xcCnbRqNmfE_-XDGK75XyiFWyP_XzzHX_SUn812LEoh7eD_WDoi-anOEfTmzQYlmqGw0SgxTO08LGWKOw2IvHVB4xWrd6Y7SbrJLxzQ4X_22jJvXSnmPB5OHNj-4s5uBxEaarmH9dij65NfCg"
    }
    
    payload = {
        "refer_instrumental": "instrumental-2025041211322025-9J7whSMp",
        "model": "music-01",
        "audio_setting": {
            "sample_rate": 44100,
            "bitrate": 256000,
            "format": "mp3"
        }
    }
    
    try:
        print("正在发送请求到API...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("API响应状态码:", response.status_code)
        
        # 检查API返回状态
        result = response.json()
        if "base_resp" not in result or result["base_resp"].get("status_code") != 0:
            print(f"API错误: {result.get('base_resp', {}).get('status_msg', '未知错误')}")
            return None
            
        return result
    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
        return None

def save_audio(audio_data, filename="output.mp3"):
    try:
        # 获取当前工作目录的绝对路径
        current_dir = os.path.abspath(os.getcwd())
        file_path = os.path.join(current_dir, filename)
        
        print("开始解码音频数据...")
        print(f"原始数据长度: {len(audio_data)}")
        
        # 验证音频数据
        if not audio_data or not isinstance(audio_data, str):
            print("无效的音频数据格式")
            return False
            
        # 解码音频数据（API返回的是16进制编码）
        try:
            # 将16进制字符串转换为大写（确保格式统一）
            audio_data = audio_data.upper()
            # 使用base16解码（对应16进制编码）
            audio_bytes = base64.b16decode(audio_data)
            print("解码后的数据长度:", len(audio_bytes))
            
            # 验证解码后的数据
            if len(audio_bytes) < 100:  # 假设有效的音频文件至少100字节
                print("解码后的数据太小，可能无效")
                return False
                
        except Exception as e:
            print(f"音频数据解码错误: {e}")
            # 尝试使用base64解码（作为备选方案）
            try:
                audio_bytes = base64.b64decode(audio_data)
                print("使用base64解码成功，数据长度:", len(audio_bytes))
            except Exception as e2:
                print(f"Base64解码也失败: {e2}")
                return False
        
        # 保存为MP3文件
        print("正在保存文件...")
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        
        # 验证文件是否成功保存
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"文件已成功保存: {file_path}")
            print(f"文件大小: {file_size} 字节")
            
            # 简单验证MP3文件头
            with open(file_path, 'rb') as f:
                header = f.read(3)
                if header != b'ID3' and header[:2] != b'\xff\xfb':
                    print("警告: 文件可能不是有效的MP3格式")
                
            return True
        else:
            print("文件保存失败")
            return False
    except Exception as e:
        print(f"保存音频时出错: {e}")
        return False

def play_audio(file_path):
    try:
        if not os.path.exists(file_path):
            print("音频文件不存在")
            return False
            
        file_size = os.path.getsize(file_path)
        if file_size < 100:  # 假设有效的音频文件至少100字节
            print("音频文件太小，可能无效")
            return False
            
        print(f"正在播放音频: {file_path}")
        print(f"文件大小: {file_size} 字节")
        
        # 使用系统默认播放器播放音频
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        elif os.name == 'posix':  # macOS/Linux
            subprocess.run(['open', file_path] if os.uname().sysname == 'Darwin' else ['xdg-open', file_path])
        return True
    except Exception as e:
        print(f"播放音频时出错: {e}")
        return False

class AudioCombiner:
    def __init__(self):
        self.auth_token = "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLkvp3nhLYiLCJVc2VyTmFtZSI6IuS-neeEtiIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTEwMzE2MTE3OTQzOTE5NTgwIiwiUGhvbmUiOiIxODA2MTYxMzkzMSIsIkdyb3VwSUQiOiIxOTEwMzE2MTE3OTM1NTMwOTcyIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDQtMTIgMTE6Mjk6NDQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.PdNCgvEmgMS6wdjSNKe4MmQKtSlYMkWfFA9pSGrysU47X7UPKbAD-jwBOdHLv9kKuP8VsTOsSTS8al-rNc-M2xYx4xOiIl4rEvO6IXp9-XIEqbr6DtwDNSFoY6SRwLAMZWMQnLpAWXVqwFu6PkB4Wb9nPX3ehLjgVFk1nRQRXwMha9QozptsLrFllrI1ZnJGiBbZ8xcCnbRqNmfE_-XDGK75XyiFWyP_XzzHX_SUn812LEoh7eD_WDoi-anOEfTmzQYlmqGw0SgxTO08LGWKOw2IvHVB4xWrd6Y7SbrJLxzQ4X_22jJvXSnmPB5OHNj-4s5uBxEaarmH9dij65NfCg"
        self.uploaded_tracks = []

    def upload_track(self, file_path, track_name, purpose='instrumental'):
        """上传单个音轨并保存信息"""
        result = upload_audio(file_path, purpose=purpose)
        if result:
            track_info = {
                'name': track_name,
                'path': file_path,
                'purpose': purpose
            }
            if 'voice_id' in result:
                track_info['id'] = result['voice_id']
                track_info['type'] = 'voice'
            elif 'instrumental_id' in result:
                track_info['id'] = result['instrumental_id']
                track_info['type'] = 'instrumental'
            
            self.uploaded_tracks.append(track_info)
            print(f"成功上传音轨 '{track_name}': {track_info['id']}")
            return True
        return False

    def combine_tracks(self, output_filename="combined_output.mp3"):
        """合并所有上传的音轨，生成音乐"""
        if not self.uploaded_tracks:
            print("没有可用的音轨")
            return False

        print("\n开始合成音乐...")
        print(f"正在合成 {len(self.uploaded_tracks)} 个音轨:")
        for track in self.uploaded_tracks:
            print(f"- {track['name']} ({track['type']}): {track['id']}")

        # 构建音乐生成请求
        url = "https://api.minimax.chat/v1/music_generation"
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.auth_token
        }
        
        # 构建请求体，严格按照API文档要求
        payload = {
            "model": "music-01",  # 必填参数
            "audio_setting": {
                "sample_rate": 44100,
                "bitrate": 256000,
                "format": "mp3"
            }
        }

        # 添加音轨参数
        instrumental_tracks = [t for t in self.uploaded_tracks if t['type'] == 'instrumental']
        voice_tracks = [t for t in self.uploaded_tracks if t['type'] == 'voice']

        if instrumental_tracks:
            payload["refer_instrumental"] = instrumental_tracks[0]['id']
        
        if voice_tracks:
            payload["refer_voice"] = voice_tracks[0]['id']
            # 如果使用了refer_voice，需要提供lyrics
            payload["lyrics"] = "##\n轻轻的\n安静的\n舒缓的\n\n平和的\n宁静的\n##"

        try:
            print("\n发送合成请求...")
            print("请求参数:", json.dumps(payload, indent=2, ensure_ascii=False))
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if "base_resp" not in result or result["base_resp"].get("status_code") != 0:
                print(f"API错误: {result.get('base_resp', {}).get('status_msg', '未知错误')}")
                return False

            # 获取并保存音频数据
            audio_data = result.get("data", {}).get("audio")
            if not audio_data:
                print("API响应中没有音频数据")
                return False

            if save_audio(audio_data, output_filename):
                print(f"\n合成成功！文件已保存为: {output_filename}")
                return True
            else:
                print("保存合成文件失败")
                return False

        except Exception as e:
            print(f"合成过程出错: {e}")
            return False

class AudioMixer:
    def __init__(self):
        self.audio_tracks = []
        self.mix_settings = {
            'sample_rate': 44100,
            'bitrate': 256000,
            'format': 'wav',
            'target_rms': 0.1  # 目标RMS值，用于音量归一化
        }
    
    def normalize_audio(self, audio, sr):
        """将音频归一化到目标RMS值"""
        # 计算当前RMS
        current_rms = np.sqrt(np.mean(audio**2))
        
        # 如果当前RMS为0，返回原始音频
        if current_rms == 0:
            return audio
            
        # 计算需要的增益
        gain = self.mix_settings['target_rms'] / current_rms
        
        # 应用增益
        normalized_audio = audio * gain
        
        # 确保不会过载
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        
        return normalized_audio
    
    def add_track(self, file_path, track_name, volume=1.0, position=0):
        """添加音轨到混合列表"""
        try:
            # 加载音频文件
            audio, sr = librosa.load(file_path, sr=self.mix_settings['sample_rate'])
            
            # 音量归一化
            audio = self.normalize_audio(audio, sr)
            
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
        
        # 计算RMS能量
        rms = librosa.feature.rms(y=audio)[0]
        
        return {
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_bandwidth': np.mean(spectral_bandwidth),
            'zero_crossing_rate': np.mean(zero_crossing_rate),
            'rms': np.mean(rms),
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
                print(f"Original RMS: {track['features']['rms']:.4f}")
                
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

class NaturalLanguageAudioProcessor:
    def __init__(self):
        self.client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1/',
            api_key='6609f0ae-c28b-41b7-be8e-2406b863a4e3'
        )
        self.audio_dirs = {
            'wind': r"D:\2\风声",
            'birds': r"D:\2\鸟叫",
            'waves': r"D:\2\海浪声",
            'rain': r"D:\2\雨声"
        }
        self.mixer = AudioMixer()
        self.default_volumes = {
            'wind': 0.3,
            'birds': 0.2,
            'waves': 0.25,
            'rain': 0.25
        }

    def analyze_request(self, user_request):
        """使用Qwen-32B模型分析用户的自然语言请求"""
        prompt = f"""
        请分析以下用户请求，并返回JSON格式的分析结果：
        1. 需要的音频类型（从wind/birds/waves/rain中选择）
        2. 处理方式（mix/concat）
        3. 每个音频的音量（0-1之间，默认0.3）
        4. 特殊效果（必须包含fade_in和fade_out）

        用户请求：{user_request}

        返回格式示例：
        {{
            "audio_types": ["wind", "rain"],
            "process_type": "mix",
            "volumes": {{"wind": 0.3, "rain": 0.25}},
            "effects": ["fade_in", "fade_out"]
        }}

        请确保：
        1. 音量值在0.2-0.3之间，确保温和
        2. 必须包含fade_in和fade_out效果
        3. 返回的是有效的JSON格式。
        """
        
        print("\n正在分析用户请求...")
        response = self.client.chat.completions.create(
            model='Qwen/QwQ-32B',
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            stream=True
        )
        
        full_response = ""
        done_reasoning = False
        
        for chunk in response:
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            answer_chunk = chunk.choices[0].delta.content
            
            if reasoning_chunk != '':
                print(reasoning_chunk, end='', flush=True)
            elif answer_chunk != '':
                if not done_reasoning:
                    print('\n\n === 分析结果 ===\n')
                    done_reasoning = True
                print(answer_chunk, end='', flush=True)
                full_response += answer_chunk
        
        try:
            # 从响应中提取JSON部分
            json_start = full_response.find('{')
            json_end = full_response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = full_response[json_start:json_end]
                result = json.loads(json_str)
                
                # 确保包含淡入淡出效果
                if 'effects' not in result:
                    result['effects'] = ['fade_in', 'fade_out']
                elif 'fade_in' not in result['effects']:
                    result['effects'].append('fade_in')
                elif 'fade_out' not in result['effects']:
                    result['effects'].append('fade_out')
                    
                # 确保音量适中
                if 'volumes' in result:
                    for key in result['volumes']:
                        result['volumes'][key] = min(max(result['volumes'][key], 0.2), 0.3)
                    
                return result
            else:
                print("无法从响应中提取JSON数据")
                return None
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None

    def find_audio_files(self, audio_types):
        """根据音频类型查找对应的音频文件"""
        audio_files = {}
        for audio_type in audio_types:
            dir_path = self.audio_dirs.get(audio_type)
            if dir_path and os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.mp3')]
                if files:
                    audio_files[audio_type] = os.path.join(dir_path, files[0])
        return audio_files

    def process_audio(self, analysis_result, output_filename="custom_mix.mp3"):
        """根据分析结果处理音频"""
        if not analysis_result:
            print("分析结果无效")
            return False
            
        audio_files = self.find_audio_files(analysis_result['audio_types'])
        
        if not audio_files:
            print("未找到匹配的音频文件")
            return False

        # 生成描述性文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        process_type = analysis_result['process_type']  # mix 或 concat
        audio_types = '_'.join(analysis_result['audio_types'])
        volumes = '_'.join([f"{k}{v:.2f}" for k, v in analysis_result['volumes'].items()])
        output_filename = os.path.join('mixed', f'{process_type}_{audio_types}_{volumes}_{timestamp}.mp3')

        if analysis_result['process_type'] == 'mix':
            # 使用AudioMixer进行混合
            for audio_type, file_path in audio_files.items():
                volume = analysis_result['volumes'].get(audio_type, self.default_volumes[audio_type])
                self.mixer.add_track(file_path, audio_type, volume)
            
            # 设置混合风格
            mix_style = 'dynamic'  # 可以根据分析结果选择不同的风格
            return self.mixer.generate_mix(output_filename, mix_style=mix_style)
        
        else:  # concat
            # 使用AudioMixer进行拼接
            for audio_type, file_path in audio_files.items():
                volume = analysis_result['volumes'].get(audio_type, self.default_volumes[audio_type])
                self.mixer.add_track(file_path, audio_type, volume)
            
            return self.mixer.generate_mix(output_filename)

    def process_natural_language_request(self, user_request):
        """处理用户的自然语言请求"""
        try:
            # 分析用户请求
            analysis_result = self.analyze_request(user_request)
            
            if not analysis_result:
                print("分析失败，请重试")
                return False
            
            # 处理音频
            print("\n处理音频...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join('mixed', f'custom_audio_{timestamp}.mp3')
            
            if self.process_audio(analysis_result, output_file):
                print(f"\n处理成功！文件已保存为: {output_file}")
                return True
            else:
                print("处理失败")
                return False
                
        except Exception as e:
            print(f"处理过程中出错: {e}")
            return False

def process_natural_language_audio():
    """处理自然语言音频请求的主函数"""
    processor = NaturalLanguageAudioProcessor()
    
    while True:
        print("\n请输入您想要的白噪音组合描述（输入'quit'退出）:")
        print("示例：")
        print("- 我想要一个轻柔的风声和雨声的混合，适合睡眠")
        print("- 请把鸟叫声和海浪声拼接在一起，做成一个自然环境的背景音")
        print("- 混合风声和雨声，风声要大声一点，雨声要轻柔一些")
        
        user_request = input("> ")
        
        if user_request.lower() == 'quit':
            break
            
        processor.process_natural_language_request(user_request)

def main():
    # 创建音频混合器实例
    mixer = AudioMixer()
    
    # 定义要合成的音轨
    tracks = [
        {"path": r"D:\2\风吹树叶.mp3", "name": "Wind", "volume": 0.3},
        {"path": r"D:\2\鸟叫.mp3", "name": "Birds", "volume": 0.2},
        {"path": r"D:\2\雨声.mp3", "name": "Rain", "volume": 0.25}
    ]
    
    # 添加所有音轨
    print("开始上传音轨...")
    for track in tracks:
        if os.path.exists(track["path"]):
            if not mixer.add_track(track["path"], track["name"], track["volume"]):
                print(f"上传音轨 '{track['name']}' 失败")
                return
        else:
            print(f"音轨文件不存在: {track['path']}")
            return

    # 生成混合音频
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    process_type = 'mix'  # 或 'concat'，取决于处理方式
    track_names = '_'.join([track["name"] for track in tracks])
    volumes = '_'.join([f"{track['name']}{track['volume']:.2f}" for track in tracks])
    output_file = os.path.join('mixed', f'{process_type}_{track_names}_{volumes}_{timestamp}.wav')
    
    # 选择混合风格
    mix_style = 'natural'  # 可选: 'natural', 'ambient', 'dynamic'
    
    if mixer.generate_mix(output_file, mix_style):
        print("\nAudio mixing completed successfully!")
        print(f"Output file: {output_file}")
    else:
        print("\nFailed to mix audio files")

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('mixed', exist_ok=True)
    
    # 添加新的入口点
    print("请选择操作模式:")
    print("1. 使用自然语言描述生成白噪音")
    print("2. 使用预设音轨组合")
    
    choice = input("请输入选项（1或2）: ")
    
    if choice == "1":
        process_natural_language_audio()
    elif choice == "2":
        main()
    else:
        print("无效的选项") 