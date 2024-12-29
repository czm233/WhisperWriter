import whisper
from deep_translator import GoogleTranslator
import pyaudio
import numpy as np
import time
import wave

# 加载 Whisper 模型
model = whisper.load_model("large")

# 设置音频参数
FORMAT = pyaudio.paInt16  # 16-bit 音频格式
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率 16kHz
CHUNK = 1024  # 每次读取的音频块大小
RECORD_SECONDS = 5  # 每次录音的时长（秒）

# 初始化 PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("开始实时语音翻译...")

try:
    while True:
        print("录音中...")
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # 将音频数据保存为临时文件
        temp_file = "temp.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        # 语音识别
        print("识别中")
        result = model.transcribe(temp_file, language="zh")
        recognized_text = result["text"]

        if recognized_text.strip():  # 如果识别到文本
            # 机器翻译
            # translated_text = translator.translate(recognized_text)

            # 打印结果
            print(f"识别结果: {recognized_text}")
#             print(f"翻译: {translated_text}")
            print("-" * 40)

        time.sleep(0.1)  # 控制循环速度

except KeyboardInterrupt:
    print("停止实时语音翻译。")

# 关闭音频流
stream.stop_stream()
stream.close()
p.terminate()