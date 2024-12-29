import pyaudio
import numpy as np

# 设置音频参数
FORMAT = pyaudio.paInt16  # 16-bit 音频格式
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率 16kHz
CHUNK = 1024              # 每次读取的音频块大小

# 初始化 PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("开始录音...")

try:
    while True:
        # 读取音频数据
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        print(f"音频数据: {audio_data}")  # 打印音频数据
except KeyboardInterrupt:
    print("停止录音。")

# 关闭音频流
stream.stop_stream()
stream.close()
p.terminate()

# import whisper
#
# # 加载 Whisper 模型
# model = whisper.load_model("base")
#
# # 测试音频文件
# audio_file = "test.mp3"  # 替换为你的音频文件路径
# result = model.transcribe(audio_file)
#
# print("识别结果:", result["text"])