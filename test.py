import threading
import queue
import pyaudio
import wave
import io
import time
import whisper
import torch
import numpy as np
import logging
import webrtcvad  # 语音活动检测
import librosa    # 如需重采样
import sys

logging.basicConfig(
    filename='transcription_errors.log',
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

global_initial_prompt = "这段音频可能包含技术术语，如 Vue.js、JavaScript、Node.js、Vue、React、Django、Drf、Python等等"

# ========== 1. Whisper 模型初始化 ==========
def init_whisper_model(model_name="medium", device=None):
    """
    初始化 Whisper 模型。
    :param model_name: Whisper 模型名称 ('tiny','base','small','medium','large')
    :param device: 'cuda' 或 'cpu'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return whisper.load_model(model_name).to(device)

# ========== 2. WAV 转 float32 波形 (Whisper 输入) ==========
def wav_bytes_to_float32(audio_bytes):
    """
    将包含 WAV 格式音频的字节流解析为 float32 的 NumPy 数组。
    返回值: (audio_float32, sample_rate)
    """
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(num_frames)

    if sample_width == 2:
        audio_np = np.frombuffer(raw_data, dtype=np.int16)
        audio_float32 = audio_np.astype(np.float32) / 32768.0
    else:
        raise ValueError("仅支持 16-bit PCM 音频")

    # 多声道时，只取第一个声道
    if num_channels > 1:
        audio_float32 = audio_float32[0::num_channels]

    return audio_float32, framerate

# ========== 4. 录音线程：基于 VAD 判断后再入队列 ==========
def record_audio(p,
                audio_queue,
                stop_event,
                rate=16000,
                chunk=480,  # 30ms的帧（16000 * 0.03）
                channels=1,
                format_=pyaudio.paInt16,
                vad_mode=3,
                silence_duration_threshold=0.5):
    """
    持续录音，基于 VAD 判断语音活动，
    当检测到语音结束时，将语音片段提交到队列。
    """
    stream = p.open(format=format_,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    vad = webrtcvad.Vad(mode=vad_mode)
    print("录音线程已启动。")

    frames = []
    is_speech = False
    silence_duration = 0.0
    frame_duration = chunk / rate  # 单帧时长

    try:
        while not stop_event.is_set():
            data = stream.read(chunk, exception_on_overflow=False)
            if vad.is_speech(data, rate):
                if not is_speech:
                    print("检测到语音开始。")
                is_speech = True
                frames.append(data)
                silence_duration = 0.0
            else:
                if is_speech:
                    silence_duration += frame_duration
                    frames.append(data)
                    if silence_duration >= silence_duration_threshold:
                        # 认为语音结束，提交音频
                        print("检测到语音结束。提交音频。")
                        # 将收集的 PCM 数据写入 WAV 缓冲区
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, 'wb') as wf:
                            wf.setnchannels(channels)
                            wf.setsampwidth(p.get_sample_size(format_))
                            wf.setframerate(rate)
                            wf.writeframes(b''.join(frames))
                        audio_data = wav_buffer.getvalue()
                        audio_queue.put(audio_data)
                        frames = []
                        is_speech = False
                        silence_duration = 0.0
                else:
                    # 静音状态，忽略
                    pass
    except Exception as e:
        logging.error(f"录音线程出现错误: {e}")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        print("录音线程已停止。")

# ========== 5. 转录线程：从队列取音频，用 Whisper 识别 ==========
def transcribe_audio(model, audio_queue, stop_event, language="zh", fp16=False):
    """
    转录线程：从 audio_queue 中获取音频数据并调用 Whisper。
    """
    print("转录线程已启动。")
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.5)  # 等待音频
        except queue.Empty:
            continue

        try:
            audio_array, sample_rate = wav_bytes_to_float32(audio_data)
            # 如果采样率不是 16000，需要重采样
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

            # Whisper 期望的输入是 float32 的音频数组，采样率为 16000 Hz
            # 确保音频片段符合要求
            result = model.transcribe(audio_array, language=language, fp16=fp16, initial_prompt=global_initial_prompt)
            text = result["text"].strip()
            if text:
                print(f"识别结果: {text}")
                print("-" * 40)
            else:
                print("未识别到有效文本。")
        except Exception as e:
            logging.error(f"转录线程出现错误: {e}")
            print("转录过程中出现错误，请查看日志以获取详细信息。")
    print("转录线程已停止。")

# ========== 6. 主函数：启动录音线程、转录线程 ==========
def main():
    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    # 初始化 Whisper 模型（可根据需求改成 "tiny", "small", "medium", "large"）
    model = init_whisper_model("turbo")

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # 录音线程
    recording_thread = threading.Thread(
        target=record_audio,
        args=(
            p,
            audio_queue,
            stop_event,
        ),
        daemon=True
    )
    recording_thread.start()

    # 转录线程
    transcription_thread = threading.Thread(
        target=transcribe_audio,
        args=(model, audio_queue, stop_event),
        daemon=True
    )
    transcription_thread.start()

    print("开始实时语音转录（基于 VAD）...按 Ctrl+C 结束。")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("停止实时语音转录。")
        stop_event.set()
    finally:
        recording_thread.join(timeout=2)
        transcription_thread.join(timeout=2)
        p.terminate()
        print("主线程已退出。")

if __name__ == "__main__":
    main()
