import os

import audioread
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
# from edge_tts_service import sync_text_to_speech, EdgeTTSConfig


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
            'format': 'wav'
        }

    def add_track(self, file_path, track_name, volume=1.0):

        # try:
        #     with audioread.audio_open(file_path) as f:
        #         print(f"Duration: {f.duration:.2f}s, Channels: {f.channels}, Samplerate: {f.samplerate}")
        # except Exception as e:
        #     print("Cannot open file:", e)
        #
        # return

        """添加音轨到混合列表"""
        try:
            print(f"正在加载音频文件: {file_path}")
            print(self.mix_settings['sample_rate'])
            # 使用 librosa 加载音频
            audio, sr = librosa.load(
                file_path,
                sr=self.mix_settings['sample_rate'],
                mono=True  # 确保是单声道
            )

            print(f"成功加载音频: {track_name}, 采样率: {sr}, 长度: {len(audio)}")

            # 应用音量
            audio = audio * volume

            self.audio_tracks.append({
                'name': track_name,
                'audio': audio,
                'volume': volume
            })
            print(f"成功添加音轨: {track_name}")
            return True
        except Exception as e:
            print(f"添加音轨失败 {track_name}: {str(e)}")
            return False

    def generate_mix(self, output_path, mix_style='natural'):
        """生成混合音频"""
        try:
            if not self.audio_tracks:
                print("没有可用的音轨")
                return False

            print("\n开始混合音频...")
            print(f"混合风格: {mix_style}")

            # 找到最长的音频长度
            max_length = max(len(track['audio']) for track in self.audio_tracks)
            mixed_audio = np.zeros(max_length)

            # 简单地将所有音轨相加
            for track in self.audio_tracks:
                print(f"\n处理音轨: {track['name']}")
                audio = track['audio']

                # 如果音频长度不足，用零填充
                if len(audio) < max_length:
                    audio = np.pad(audio, (0, max_length - len(audio)))

                # 直接相加
                mixed_audio += audio

            # 防止削波
            max_value = np.max(np.abs(mixed_audio))
            if max_value > 1.0:
                mixed_audio = mixed_audio / max_value

            # 保存音频
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
                audio_files[audio_type] = "audio/喜马拉雅风声.wav"
            elif audio_type == 'rain':
                audio_files[audio_type] = "audio/喜马拉雅雨声.wav"
            elif audio_type == 'birds':
                audio_files[audio_type] = "audio/鸟叫.wav"
            elif audio_type == 'waves':
                audio_files[audio_type] = "audio/hrxz.com-cn22iesbwcz45727.wav"

        # 验证文件是否存在
        for audio_type, file_path in list(audio_files.items()):
            if not os.path.exists(file_path):
                print(f"音频文件不存在: {file_path}")
                # 尝试其他可能的文件名
                alt_paths = [
                    file_path,
                    file_path.replace('.mp3', '.wav'),
                    file_path.replace('喜马拉雅', ''),
                    os.path.join('audio', f'{audio_type}.wav')
                ]

                found = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        print(f"找到替代文件: {alt_path}")
                        audio_files[audio_type] = alt_path
                        found = True
                        break

                if not found:
                    print(f"未找到任何可用的音频文件")
                    del audio_files[audio_type]
            else:
                print(f"找到音频文件: {file_path}")

        return audio_files

    def generate_prompt_voice(self, text, output_path):
        """使用 Google TTS 生成提示语音"""
        try:
            # 使用 gTTS 生成语音
            tts = gTTS(text=text, lang='zh-cn', slow=False)
            tts.save(output_path)
            print(f"提示语音已生成: {output_path}")
            return True
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
            # 使用 numpy 实现淡入淡出效果
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)

            # 应用淡入淡出
            transition[:fade_length] *= fade_in
            transition[-fade_length:] *= fade_out
            prompt_audio[:fade_length] *= fade_in
            prompt_audio[-fade_length:] *= fade_out
            main_audio[:fade_length] *= fade_in
            main_audio[-fade_length:] *= fade_out

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
        print(f"生成音频: {output_filename}")

        # 生成处理结果提示音
        prompt_text = f"根据您的需求：{user_input}，正在生成由{' 和 '.join([self.audio_type_names.get(t, t) for t in analysis_result['audio_types']])}通过{process_type}方式合成的音频"
        prompt_file = os.path.join('audio/prompts', f'prompt_{timestamp}.mp3')
        self.generate_prompt_voice(prompt_text, prompt_file)

        # 根据处理类型选择不同的处理方式
        if analysis_result['intent'] == 'mix':
            # 使用AudioMixer处理音频混合
            for audio_type, file_path in audio_files.items():
                volume = analysis_result['volumes'].get(audio_type, 0.3)
                if not self.mixer.add_track(file_path, audio_type, volume=volume):
                    print(f"添加音轨失败: {audio_type}")
                    return False

            # 生成混合音频
            if self.mixer.generate_mix(output_filename, mix_style='natural'):
                print(f"混合音频生成成功: {output_filename}")
                # 使用Stability AI生成增强版本
                enhanced_filename = os.path.join('mixed', f'enhanced_{process_type}_{audio_types}_{timestamp}.mp3')
                if self.stability_generator.generate_audio(
                        output_filename,
                        enhanced_filename,
                        prompt=f"Enhance the {process_type} of {', '.join(analysis_result['audio_types'])} sounds",
                        duration=180
                ):
                    print(f"增强版本生成成功: {enhanced_filename}")
                    # 添加提示音到最终音频
                    final_audio = self.add_audio_prompt(
                        enhanced_filename,
                        analysis_result['audio_types'],
                        process_type,
                        audio_files,
                        prompt_file
                    )
                    return final_audio
            else:
                print("混合音频生成失败")

        elif analysis_result['intent'] == 'concat':
            # 使用简单的音频拼接
            try:
                # 加载所有音频文件
                audio_segments = []
                for audio_type, file_path in audio_files.items():
                    print(f"正在加载音频文件: {file_path}")
                    audio, sr = librosa.load(file_path, sr=44100, mono=True)
                    audio_segments.append(audio)

                # 添加过渡音效
                transition, _ = librosa.load(self.transition_sound)

                # 拼接所有音频段
                final_audio = []
                for i, segment in enumerate(audio_segments):
                    # 添加音频段
                    final_audio.append(segment)
                    # 如果不是最后一个音频段，添加过渡音效
                    if i < len(audio_segments) - 1:
                        final_audio.append(transition)

                # 合并所有音频段
                final_audio = np.concatenate(final_audio)

                # 保存拼接后的音频
                sf.write(output_filename, final_audio, sr)
                print(f"拼接音频生成成功: {output_filename}")

                # 添加提示音到最终音频
                final_audio = self.add_audio_prompt(
                    output_filename,
                    analysis_result['audio_types'],
                    process_type,
                    audio_files,
                    prompt_file
                )
                return final_audio
            except Exception as e:
                print(f"拼接音频时出错: {str(e)}")
                return False

        elif analysis_result['intent'] == 'fusion':
            # 使用StabilityAudioGenerator处理音频融合
            try:
                # 使用第一个音频作为基础
                base_audio = list(audio_files.values())[0]

                # 使用Stability AI生成融合音频
                if self.stability_generator.generate_audio(
                        base_audio,
                        output_filename,
                        prompt=f"Fuse {', '.join(analysis_result['audio_types'])} sounds into a cohesive audio experience",
                        duration=180
                ):
                    print(f"融合音频生成成功: {output_filename}")
                    # 添加提示音到最终音频
                    final_audio = self.add_audio_prompt(
                        output_filename,
                        analysis_result['audio_types'],
                        process_type,
                        audio_files,
                        prompt_file
                    )
                    return final_audio
                else:
                    print("融合音频生成失败")
                    return False
            except Exception as e:
                print(f"融合音频时出错: {str(e)}")
                return False

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