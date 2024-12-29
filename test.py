import whisper
import torch
import pyaudio
import numpy as np
import wave
import io
import time
import threading
import queue
import logging

# 配置日志
logging.basicConfig(filename='transcription_errors.log',
                    level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# ===========================
# 1. 初始化 Whisper 模型
# ===========================

def init_whisper_model(model_name="large", device=None):
    """
    初始化 Whisper 模型。
    :param model_name: Whisper 模型名称，如 'large', 'medium', 'small' 等
    :param device: 指定使用的设备 'cuda' 或 'cpu'
    :return: 加载好的 whisper 模型
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = whisper.load_model(model_name).to(device)
    return model

# ===========================
# 2. 录音函数
# ===========================

def record_audio(p, audio_queue, record_seconds=2, rate=16000, chunk=1024, channels=1, format_=pyaudio.paInt16):
    """
    持续录制音频，并将录制的音频块放入队列中。
    """
    stream = p.open(format=format_,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    print("录音线程已启动。")
    try:
        while True:
            frames = []
            for _ in range(0, int(rate / chunk * record_seconds)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
            # 将录音数据转换为 WAV 字节流
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(format_))
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
            audio_data = wav_buffer.getvalue()
            # 将音频数据放入队列
            audio_queue.put(audio_data)
    except Exception as e:
        logging.error(f"录音线程出现错误: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        print("录音线程已停止。")

# ===========================
# 3. WAV 字节流转换为 NumPy 数组
# ===========================

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

    if num_channels > 1:
        audio_float32 = audio_float32[0::num_channels]

    return audio_float32, framerate

# ===========================
# 4. 转录函数
# ===========================

def transcribe_audio(model, audio_queue, stop_event, language="zh", fp16=False):
    """
    从音频队列中获取音频数据并进行转录。
    """
    print("转录线程已启动。")
    while not stop_event.is_set():
        try:
            # 尝试获取音频数据，超时可避免阻塞退出
            audio_data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            audio_array, sample_rate = wav_bytes_to_float32(audio_data)
            # Whisper expects audio to be 16000 Hz; resample if necessary
            if sample_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            result = model.transcribe(audio_array, language=language, fp16=fp16)
            text = result["text"].strip()

            if text:
                print(f"识别结果: {text}")
                print("-" * 40)
            else:
                print("未识别到有效文本。")
        except Exception as e:
            logging.error(f"转录线程出现错误: {e}")
            print("转录过程中出现错误，检查日志以获取详细信息。")
    print("转录线程已停止。")

# ===========================
# 5. 主函数
# ===========================

def main():
    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 初始化 Whisper 模型
    model = init_whisper_model(model_name="large")  # 可以根据需求调整模型大小

    # 创建一个线程安全的队列用于传递音频数据
    audio_queue = queue.Queue()

    # 创建一个事件用于停止线程
    stop_event = threading.Event()

    # 启动录音线程
    recording_thread = threading.Thread(target=record_audio, args=(p, audio_queue), daemon=True)
    recording_thread.start()

    # 启动转录线程
    transcription_thread = threading.Thread(target=transcribe_audio, args=(model, audio_queue, stop_event), daemon=True)
    transcription_thread.start()

    print("开始实时语音翻译...按 Ctrl+C 结束。")
    try:
        while True:
            time.sleep(0.1)  # 主线程保持活跃
    except KeyboardInterrupt:
        print("停止实时语音翻译。")
        stop_event.set()
    finally:
        # 等待线程结束
        recording_thread.join(timeout=1)
        transcription_thread.join(timeout=1)
        p.terminate()

if __name__ == "__main__":
    main()
