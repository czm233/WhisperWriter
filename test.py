import whisper
import torch
import pyaudio
import numpy as np
import wave
import io
import time

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

def record_audio(p, record_seconds=3, rate=16000, chunk=1024, channels=1, format_=pyaudio.paInt16):
    """
    录制音频数据并返回字节流（wav 格式）。
    """
    stream = p.open(format=format_,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    print("录音中...")
    frames = []
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()

    # 将录音数据写入 BytesIO，形成一个合法的 WAV 文件字节流
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format_))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    return wav_buffer.getvalue()

def wav_bytes_to_float32(audio_bytes):
    """
    将包含 WAV 格式音频的字节流解析为 float32 的 NumPy 数组。
    返回值: (audio_float32, sample_rate)
    """
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()  # 一般是 2 表示 16-bit
        framerate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(num_frames)

    # 根据声道数与位宽将数据转换为 numpy 数组
    if sample_width == 2:
        audio_np = np.frombuffer(raw_data, dtype=np.int16)  # 16-bit PCM
        # 转为 float32，并缩放到 [-1, 1]
        audio_float32 = audio_np.astype(np.float32) / 32768.0
    else:
        # 如果有其他格式，需根据实际情况做更多判断
        raise ValueError("仅示例支持 16-bit PCM 音频")

    # 如果有多声道，可只取第一声道或做混合
    if num_channels > 1:
        # 例如只取第一个声道
        audio_float32 = audio_float32[0::num_channels]

    return audio_float32, framerate

def transcribe_audio(model, audio_data, language="zh", fp16=False):
    """
    使用 Whisper 模型转写给定音频（在内存中的 WAV 字节）。
    """
    # 1) 将 WAV 字节流转换为 float32 waveform
    audio_array, sample_rate = wav_bytes_to_float32(audio_data)

    # 2) 调用 whisper transcribe
    #    如果在 CPU 上运行或显存不足，可将 fp16=False
    result = model.transcribe(audio_array, language=language, fp16=fp16)
    return result["text"]

def main():
    # 1. 初始化 PyAudio 和模型
    p = pyaudio.PyAudio()
    model = init_whisper_model(model_name="large")  # 或 "medium"/"small"/"tiny" 以加快速度

    print("开始实时语音翻译...按 Ctrl+C 结束。")
    try:
        while True:
            # 2. 录音并获取内存中的 wav 字节流
            audio_data = record_audio(
                p,
                record_seconds=3,
                rate=16000,
                chunk=1024,
                channels=1,
                format_=pyaudio.paInt16
            )

            # 3. 转写
            print("识别中...")
            recognized_text = transcribe_audio(model, audio_data, language="zh", fp16=False)

            # 4. 打印结果
            if recognized_text.strip():
                print(f"识别结果: {recognized_text}")
                print("-" * 40)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("停止实时语音翻译。")
    finally:
        p.terminate()

if __name__ == "__main__":
    main()
