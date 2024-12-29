# WhisperWriter
实时语音转文字
使用large效果会更好
model = whisper.load_model("large")
用gpu推理更快
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
注意pip install openai-whisper而不是whisper
注意录音设备录音的默认格式