import os
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import json
import requests
from pathlib import Path
from gtts import gTTS
import time
import subprocess
from save_audio import AudioCombiner
from music_generation import StabilityAudioGenerator
from edge_tts_service import sync_text_to_speech, EdgeTTSConfig

class IntentAnalyzer:
    def __init__(self):
        self.keywords = {
            'mix': ['混合', '混音', '一起', '同时'],
            'concat': ['拼接', '连接', '接在一起', '先后'],
            'fusion': ['融合', '合成', '生成', '创作']
        }
        
        # 音频类型关键词
        self.audio_types = {
            'wind': ['风', '风声', '风吹'],
            'rain': ['雨', '雨声', '下雨'],
            'birds': ['鸟', '鸟叫', '鸟鸣'],
            'waves': ['海', '海浪', '海声']
        }
    
    def analyze(self, user_input):
        """分析用户输入，返回意图和音频类型"""
        result = {
            'intent': None,
            'audio_types': [],
            'volumes': {}
        }
        
        # 分析处理意图
        for key, words in self.keywords.items():
            if any(word in user_input for word in words):
                result['intent'] = key
                break
        
        # 分析音频类型
        for audio_type, words in self.audio_types.items():
            if any(word in user_input for word in words):
                result['audio_types'].append(audio_type)
                # 设置默认音量
                result['volumes'][audio_type] = 0.3
        
        # 分析音量关键词
        volume_keywords = {
            '大': 0.5,
            '小': 0.2,
            '中': 0.3
        }
        
        for word, volume in volume_keywords.items():
            for audio_type in result['audio_types']:
                if f"{audio_type}{word}" in user_input:
                    result['volumes'][audio_type] = volume
        
        return result

class AudioMixer:
    def __init__(self):
        self.audio_tracks = []
        self.mix_settings = {
            'sample_rate': 44100,
            'bitrate': 256000,
            'format': 'wav',
            'target_rms': 0.15,  # 提高目标RMS值，使声音更饱满
            'compression_threshold': 0.6,  # 提高压缩阈值
            'compression_ratio': 1.8  # 降低压缩比，保留更多动态范围
        }
    
    def normalize_audio(self, audio, sr):
        """将音频归一化到目标RMS值，并添加动态范围压缩"""
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms == 0:
            return audio
        
        # 应用动态范围压缩
        threshold = self.mix_settings['compression_threshold']
        ratio = self.mix_settings['compression_ratio']
        compressed_audio = np.copy(audio)
        mask = np.abs(audio) > threshold
        compressed_audio[mask] = threshold + (np.abs(audio[mask]) - threshold) / ratio
        compressed_audio = np.sign(audio) * compressed_audio
        
        # 添加轻微的谐波失真，使声音更温暖
        harmonic_distortion = 0.05
        compressed_audio = compressed_audio + harmonic_distortion * np.tanh(compressed_audio)
        
        # 归一化
        gain = self.mix_settings['target_rms'] / np.sqrt(np.mean(compressed_audio**2))
        normalized_audio = compressed_audio * gain
        return np.clip(normalized_audio, -1.0, 1.0)
    
    def add_track(self, file_path, track_name, volume=1.0):
        """添加音轨到混合列表，并应用随机相位偏移"""
        try:
            audio, sr = librosa.load(file_path, sr=self.mix_settings['sample_rate'])
            
            # 添加随机相位偏移
            phase_shift = np.random.random() * 2 * np.pi
            audio = np.roll(audio, int(phase_shift * sr / (2 * np.pi)))
            
            # 应用更平滑的淡入淡出
            fade_length = int(0.15 * sr)  # 增加淡入淡出时间
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            audio[:fade_length] *= fade_in
            audio[-fade_length:] *= fade_out
            
            # 添加轻微的随机音量波动
            volume_fluctuation = 0.1
            fluctuation = 1 + volume_fluctuation * np.sin(np.linspace(0, 2*np.pi, len(audio)))
            audio *= fluctuation
            
            audio = self.normalize_audio(audio, sr)
            
            self.audio_tracks.append({
                'name': track_name,
                'audio': audio,
                'volume': volume
            })
            print(f"添加音轨: {track_name}")
            return True
        except Exception as e:
            print(f"添加音轨失败 {track_name}: {str(e)}")
            return False
    
    def generate_mix(self, output_path, mix_style='natural'):
        """生成混合音频，使用更自然的混合方式"""
        try:
            if not self.audio_tracks:
                print("没有可用的音轨")
                return False
            
            print("\n开始混合音频...")
            print(f"混合风格: {mix_style}")
            
            max_length = max(len(track['audio']) for track in self.audio_tracks)
            mixed_audio = np.zeros(max_length)
            
            for track in self.audio_tracks:
                print(f"\n处理音轨: {track['name']}")
                audio = track['audio']
                
                if len(audio) < max_length:
                    # 使用循环填充而不是零填充
                    repeats = int(np.ceil(max_length / len(audio)))
                    audio = np.tile(audio, repeats)[:max_length]
                else:
                    audio = audio[:max_length]
                
                # 应用更自然的音量渐变
                volume_curve = np.linspace(0.9, 1.1, max_length)
                volume_curve += 0.05 * np.sin(np.linspace(0, 4*np.pi, max_length))  # 添加轻微波动
                audio = audio * track['volume'] * volume_curve
                
                mixed_audio += audio
            
            # 应用最终动态范围压缩
            mixed_audio = self.normalize_audio(mixed_audio, self.mix_settings['sample_rate'])
            
            # 添加更自然的混响效果
            reverb_length = int(0.15 * self.mix_settings['sample_rate'])  # 增加混响时间
            reverb = np.exp(-np.linspace(0, 4, reverb_length))  # 更平滑的衰减
            mixed_audio = np.convolve(mixed_audio, reverb, mode='same')
            
            # 添加轻微的立体声效果
            stereo_width = 0.2
            left = mixed_audio * (1 - stereo_width)
            right = mixed_audio * (1 + stereo_width)
            mixed_audio = np.vstack((left, right)).T
            
            sf.write(output_path, mixed_audio, self.mix_settings['sample_rate'])
            print(f"\n混合成功！文件已保存为: {output_path}")
            return True
        except Exception as e:
            print(f"混合音频时出错: {str(e)}")
            return False

class AudioProcessor:
    def __init__(self):
        self.audio_dirs = {
            'wind': 'audio/wind',
            'rain': 'audio/rain',
            'birds': 'audio/birds',
            'waves': 'audio/waves'
        }
        self.audio_type_names = {
            'wind': '风声',
            'rain': '雨声',
            'birds': '鸟叫声',
            'waves': '海浪声'
        }
        self.transition_sound = "audio/transition.wav"
        self.mixer = AudioMixer()
        self.stability_generator = StabilityAudioGenerator()
        
        # 创建必要的目录
        for dir_path in self.audio_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        os.makedirs('mixed', exist_ok=True)
        os.makedirs('audio/prompts', exist_ok=True)
    
    def find_audio_files(self, audio_types):
        """根据音频类型查找对应的音频文件"""
        audio_files = {}
        for audio_type in audio_types:
            # 根据音频类型选择对应的文件
            if audio_type == 'wind':
                audio_files[audio_type] = "audio/喜马拉雅风声.mp3"
            elif audio_type == 'rain':
                audio_files[audio_type] = "audio/喜马拉雅雨声.mp3"
            elif audio_type == 'birds':
                audio_files[audio_type] = "audio/鸟叫.wav"
            elif audio_type == 'waves':
                audio_files[audio_type] = "audio/hrxz.com-cn22iesbwcz45727.wav"
        
        # 验证文件是否存在
        for audio_type, file_path in list(audio_files.items()):
            if not os.path.exists(file_path):
                print(f"音频文件不存在: {file_path}")
                del audio_files[audio_type]
        
        return audio_files
    
    def generate_prompt_voice(self, text, output_path):
        """使用 Edge TTS 生成提示语音"""
        try:
            # 配置 TTS 参数
            config = EdgeTTSConfig(
                voice="zh-CN-XiaoyiNeural",  # 使用小艺的声音
                rate="+0%",      # 正常语速
                volume="+20%",   # 音量增加20%
                pitch="+0%"      # 正常音调
            )
            
            # 生成语音
            if sync_text_to_speech(text, output_path, config):
                print(f"提示语音已生成: {output_path}")
                return True
            else:
                print("生成提示语音失败")
                return False
        except Exception as e:
            print(f"生成提示语音失败: {str(e)}")
            return False
    
    def add_audio_prompt(self, audio_path, source_tracks, process_type, audio_files, prompt_file):
        """添加提示音和语音说明"""
        try:
            # 加载音频文件
            transition, sr = librosa.load(self.transition_sound)
            prompt_audio, _ = librosa.load(prompt_file)
            main_audio, _ = librosa.load(audio_path)
            
            # 添加淡入淡出效果
            fade_length = int(0.1 * sr)
            transition = librosa.effects.fade(transition, fade_length)
            prompt_audio = librosa.effects.fade(prompt_audio, fade_length)
            main_audio = librosa.effects.fade(main_audio, fade_length)
            
            # 拼接音频
            final_audio = np.concatenate([
                transition,
                prompt_audio,
                transition,
                main_audio
            ])
            
            # 保存最终音频
            output_path = f"prompted_{os.path.basename(audio_path)}"
            sf.write(output_path, final_audio, sr)
            
            # 清理临时文件
            os.remove(prompt_file)
            
            return output_path
        except Exception as e:
            print(f"添加提示音时出错: {str(e)}")
            return None

    def process_audio(self, analysis_result, user_input):
        """根据分析结果处理音频"""
        if not analysis_result or not analysis_result['audio_types']:
            print("分析结果无效")
            return False
        
        audio_files = self.find_audio_files(analysis_result['audio_types'])
        if not audio_files:
            print("未找到匹配的音频文件")
            return False
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        process_type = analysis_result['intent']
        audio_types = '_'.join(analysis_result['audio_types'])
        output_filename = os.path.join('mixed', f'{process_type}_{audio_types}_{timestamp}.wav')
        
        # 根据处理类型选择不同的处理方式
        if analysis_result['intent'] == 'mix':
            # 使用AudioMixer处理音频
            for audio_type, file_path in audio_files.items():
                volume = analysis_result['volumes'].get(audio_type, 0.3)
                if not self.mixer.add_track(file_path, audio_type, volume=volume):
                    print(f"添加音轨失败: {audio_type}")
                    return False
            
            # 生成混合音频
            if self.mixer.generate_mix(output_filename, mix_style='natural'):
                # 使用Stability AI生成增强版本
                enhanced_filename = os.path.join('mixed', f'enhanced_{process_type}_{audio_types}_{timestamp}.mp3')
                if self.stability_generator.generate_audio(
                    output_filename,
                    enhanced_filename,
                    prompt=f"Enhance the {process_type} of {', '.join(analysis_result['audio_types'])} sounds",
                    duration=180
                ):
                    # 生成处理结果提示音
                    prompt_text = f"根据您的需求：{user_input}，已生成由{' 和 '.join([self.audio_type_names.get(t, t) for t in analysis_result['audio_types']])}通过{process_type}方式合成的音频"
                    prompt_file = os.path.join('audio/prompts', f'result_{timestamp}.mp3')
                    self.generate_prompt_voice(prompt_text, prompt_file)
                    
                    # 添加提示音到最终音频
                    final_audio = self.add_audio_prompt(
                        enhanced_filename,
                        analysis_result['audio_types'],
                        process_type,
                        audio_files,
                        prompt_file
                    )
                    return final_audio
        elif analysis_result['intent'] == 'concat':
            # 实现拼接逻辑
            pass
        elif analysis_result['intent'] == 'fusion':
            # 实现融合逻辑
            pass
        
        return False

def play_audio(file_path):
    """播放音频文件"""
    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False
            
        print(f"\n正在播放: {file_path}")
        if os.uname().sysname == 'Darwin':  # macOS
            subprocess.run(['afplay', file_path])
        else:  # 其他系统
            subprocess.run(['xdg-open', file_path])
        return True
    except Exception as e:
        print(f"播放音频时出错: {str(e)}")
        return False

def main():
    processor = AudioProcessor()
    intent_analyzer = IntentAnalyzer()
    
    print("欢迎使用音频处理系统！")
    print("支持的音频类型：风声、雨声、鸟叫声、海浪声")
    print("支持的处理方式：混合、拼接、融合")
    print("示例：'我想要一个轻柔的风声和雨声的混合'")
    print("输入'quit'退出程序\n")
    
    # 生成欢迎提示音
    processor.generate_prompt_voice("欢迎使用音频处理系统", "audio/prompts/welcome.mp3")
    
    while True:
        user_input = input("请输入您的需求: ")
        
        if user_input.lower() == 'quit':
            processor.generate_prompt_voice("感谢使用，再见", "audio/prompts/goodbye.mp3")
            break
            
        result = intent_analyzer.analyze(user_input)
        print("\n分析结果:")
        print(f"处理方式: {result['intent']}")
        print(f"音频类型: {result['audio_types']}")
        print(f"音量设置: {result['volumes']}")
        
        if not result['audio_types']:
            processor.generate_prompt_voice("分析结果无效，请重试", "audio/prompts/invalid.mp3")
            print("分析结果无效")
            continue
            
        if not processor.process_audio(result, user_input):
            processor.generate_prompt_voice("处理失败，请重试", "audio/prompts/failed.mp3")
            print("处理失败，请重试")
            continue
            
        print("处理完成！")

if __name__ == "__main__":
    main() 