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


# ========== 2. WebRTC VAD 相关函数 ==========

def chunk_to_frames(audio_bytes, frame_duration_ms=30, sample_rate=16000):
    """
    将一段音频拆分为更小的帧（frame），每帧时长 frame_duration_ms 毫秒。
    返回值：迭代器，每次返回 (frame_data, frame_timestamp)
    """
    bytes_per_sample = 2  # 16-bit PCM
    num_channels = 1      # 单声道
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * bytes_per_sample * num_channels)

    start = 0
    timestamp = 0.0
    duration = (float(frame_size) / (bytes_per_sample * sample_rate))
    while start + frame_size <= len(audio_bytes):
        yield audio_bytes[start:start + frame_size], timestamp
        start += frame_size
        timestamp += duration

def is_speech_chunk(audio_bytes, sample_rate=16000, vad_mode=0, threshold_ratio=0.05):
    """
    对一段 PCM 音频进行 VAD 检测。如果语音帧占比超过 threshold_ratio，返回 True，否则 False。
      - vad_mode=3 表示最严格的检测，数值越大越严格（0~3）。
      - threshold_ratio 表示语音帧占总帧数的占比阈值。
    """
    vad = webrtcvad.Vad(mode=vad_mode)
    frames = list(chunk_to_frames(audio_bytes, frame_duration_ms=30, sample_rate=sample_rate))

    if not frames:
        return False

    speech_count = 0
    for frame_data, _ in frames:
        # VAD 要求每帧是 16-bit 单声道 PCM
        # 如果是 16kHz，30ms 的帧，应是 480 个采样点 * 2 字节 = 960 字节
        # webrtcvad 检测是否为语音
        if vad.is_speech(frame_data, sample_rate):
            speech_count += 1

    ratio = speech_count / len(frames)
    print(f"Debug: {speech_count}/{len(frames)} => {ratio:.2f}")  # 可用于调试
    return (ratio >= threshold_ratio)


# ========== 3. WAV 转 float32 波形 (Whisper 输入) ==========

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


# ========== 4. 录音线程：带 VAD 判断后再入队列 ==========

def record_audio(p,
                 audio_queue,
                 stop_event,
                 record_seconds=2,
                 rate=16000,
                 chunk=1024,
                 channels=1,
                 format_=pyaudio.paInt16,
                 vad_mode=3,
                 vad_ratio_threshold=0.5):
    """
    持续录音，每段时长 record_seconds 秒；
    用 WebRTC VAD 判断这段音频是否含语音，若语音占比较高，就入队列；否则丢弃。
    """
    stream = p.open(format=format_,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("录音线程已启动。")

    try:
        while not stop_event.is_set():
            frames = []
            for _ in range(0, int(rate / chunk * record_seconds)):
                if stop_event.is_set():
                    break
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

            if frames:
                # 组装成 WAV 字节流
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(p.get_sample_size(format_))
                    wf.setframerate(rate)
                    wf.writeframes(b''.join(frames))
                audio_data = wav_buffer.getvalue()

                # 在这里做 VAD 检测
                # 提取出纯 PCM（不包含 WAV header）的部分
                # wave.open() 写入的是带 WAV 头的格式，但 webrtcvad 需要原始 PCM
                # 不过为了简单，我们使用 is_speech_chunk 时，直接再次把 wav 数据去掉头部
                # 或者你也可以做一次精确的 PCM 提取
                # 这里为了示例，直接用 wav_bytes_to_float32 获取波形，然后再转回 PCM:
                audio_array, sr = wav_bytes_to_float32(audio_data)
                # 如果采样率不一致，还需要重采样到 16kHz 的 PCM
                if sr != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    sr = 16000
                # 转回 16-bit PCM (单声道)
                audio_int16 = (audio_array * 32768.0).astype(np.int16).tobytes()

                # 判断是否为语音
                if is_speech_chunk(audio_int16, sample_rate=sr,
                                   vad_mode=vad_mode,
                                   threshold_ratio=vad_ratio_threshold):
                    # 如果判定为语音数据，则放入队列
                    audio_queue.put(audio_data)
                else:
                    # 否则丢弃
                    print("【VAD】检测为非语音，丢弃该段音频...")

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

            result = model.transcribe(audio_array, language=language, fp16=fp16)
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
    model = init_whisper_model("large")

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

    print("开始实时语音翻译（带 VAD）...按 Ctrl+C 结束。")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("停止实时语音翻译。")
        stop_event.set()
    finally:
        recording_thread.join(timeout=2)
        transcription_thread.join(timeout=2)
        p.terminate()
        print("主线程已退出。")

if __name__ == "__main__":
    main()
