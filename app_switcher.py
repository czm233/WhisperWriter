import win32gui
from win10toast import ToastNotifier


def _enum_windows_callback(hwnd, windows_list):
    # 判断是否是可见窗口，同时窗口标题不为空
    if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
        windows_list.append((hwnd, win32gui.GetWindowText(hwnd)))


def switch_to_app(keyword):
    windows_list = list_all_windows()
    toaster = ToastNotifier()
    for hwnd, title in windows_list:
        # 不区分大小写匹配关键字
        if keyword.lower() in title.lower():
            try:
                # 检查窗口句柄是否有效
                if not win32gui.IsWindow(hwnd):
                    print(f"无效的窗口句柄: {hwnd}, 标题: {title}")
                    continue

                # 将窗口置顶
                win32gui.SetForegroundWindow(hwnd)
                # 显示通知
                toaster.show_toast("App Switcher", f"已切换到: {title}", duration=2)
                return True
            except Exception as e:
                print(f"切换窗口时出错: {e}")
                toaster.show_toast("App Switcher", f"切换窗口失败: {title}", duration=2)
                return False

    # 未找到匹配窗口
    toaster.show_toast("App Switcher", f"未找到包含 '{keyword}' 的窗口", duration=2)
    return False


# 枚举系统中所有可视窗口
def list_all_windows():
    windows_list = []
    win32gui.EnumWindows(_enum_windows_callback, windows_list)
    print(windows_list)
    return windows_list


if __name__ == "__main__":
    switch_to_app("微信")
