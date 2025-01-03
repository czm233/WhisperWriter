# -*- coding: utf-8 -*-
from pywinauto import Application
import time
import win32gui


def get_focused_window():
    hwnd = win32gui.GetForegroundWindow()
    return hwnd


def send_text_to_window(text, delay=1):
    handle = get_focused_window()
    time.sleep(delay)
    try:
        # 连接到目标应用程序
        app = Application().connect(handle=handle)
        window = app.window(handle=handle)

        # 设置焦点
        window.set_focus()
        time.sleep(0.1)

        # 发送文本
        window.type_keys(text, with_spaces=True, set_foreground=True)

        print(f"成功将文本 '{text}' 输出到窗口 '{handle}'。")
        print("-" * 40)
    except Exception as e:
        print(f"插入文本失败: {e}")


if __name__ == "__main__":
    # 替换为你的目标窗口标题的一部分
    text_to_output = "中文输出测试"
    send_text_to_window(text_to_output)
