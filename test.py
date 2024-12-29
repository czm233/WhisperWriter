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

logging.basicConfig(filename='transcription_errors.log',
                    level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def init_whisper_model(model_name="small", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return whisper.load_model(model_name).to(device)

def record_audio(p, audio_queue, stop_event, record_seconds=2, rate=16000, chunk=1024, channels=1, format_=pyaudio.paInt16):
    """
    持续录音，每段时长 record_seconds 秒，将得到的音频字节流放入 audio_queue。
    当 stop_event 被设置时，退出循环并停止录音。
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
                # 如果 stop_event 在这期间被设置了，就早点退出循环
                if stop_event.is_set():
                    break
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

            if frames:
                # 将录音数据转换为 WAV 字节流
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(p.get_sample_size(format_))
                    wf.setframerate(rate)
                    wf.writeframes(b''.join(frames))
                audio_data = wav_buffer.getvalue()
                audio_queue.put(audio_data)

    except Exception as e:
        logging.error(f"录音线程出现错误: {e}")

    finally:
        # 这里要检查一下流是否还打开再关闭，避免二次 close
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        print("录音线程已停止。")

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

def transcribe_audio(model, audio_queue, stop_event, language="zh", fp16=False):
    """
    转录线程：持续从 audio_queue 中获取音频数据并调用 Whisper 转写。
    """
    print("转录线程已启动。")
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue  # 如果没取到数据就重试

        try:
            audio_array, sample_rate = wav_bytes_to_float32(audio_data)
            # 如果采样率不是 16000，需要重采样
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

def main():
    p = pyaudio.PyAudio()
    model = init_whisper_model("small")

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # 录音线程
    recording_thread = threading.Thread(
        target=record_audio,
        args=(p, audio_queue, stop_event),
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

    print("开始实时语音翻译...按 Ctrl+C 结束。")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("停止实时语音翻译。")
        # 通知子线程退出
        stop_event.set()
    finally:
        # 等待线程退出
        recording_thread.join(timeout=2)
        transcription_thread.join(timeout=2)
        p.terminate()
        print("主线程已退出。")

if __name__ == "__main__":
    main()
