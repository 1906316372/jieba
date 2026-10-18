import os
import time
import ctypes
from ctypes import wintypes
import sys

# 常量定义
# 发送类型枚举
class SendType:
    AnyDriver = 0
    SendInput = 1
    Logitech = 2
    DD = 3

# 初始化标志
class InitFlags:
    NONE = 0
    NoKeyboardState = 1

# 错误代码
class Error:
    Success = 0
    DeviceNotFound = 1
    InvalidArgument = 2
    Unknown = 3

# Windows 输入结构体定义
class KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
    ]

class InputUnion(ctypes.Union):
    _fields_ = [
        ("ki", KeyboardInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("ii", InputUnion)
    ]

# 常量
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_O = 0x4F  # 'O' 键的虚拟键码
VK_I = 0x49  # 'I' 键的虚拟键码
VK_L = 0x4C  # 'L' 键的虚拟键码
VK_CONTROL = 0x11  # Ctrl键的虚拟键码
VK_0 = 0x30  # '0'键的虚拟键码

# 全局变量存储DLL和初始化状态
send_lib = None
is_initialized = False

def initialize_keyboard_simulator():
    """初始化键盘模拟器"""
    global send_lib, is_initialized
    
    if is_initialized:
        return True
        
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # DLL 路径
        dll_path = os.path.join(current_dir, "keyboardMoNiQi", "IbAhkSendLib", "v2", "IbAhkSend.dll")
        
        print(f"\n尝试加载DLL: {dll_path}")
        
        # 加载DLL
        send_lib = ctypes.CDLL(dll_path)
        print("DLL加载成功")
        
        # 设置函数返回类型和参数类型
        send_lib.IbAhkSendInit.restype = ctypes.c_int
        send_lib.IbAhkSendInit.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        
        send_lib.IbAhkSendInput.restype = ctypes.c_uint
        send_lib.IbAhkSendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(Input), ctypes.c_int]
        
        # 初始化Logitech驱动
        print("正在初始化Logitech驱动...")
        result = send_lib.IbAhkSendInit(SendType.Logitech, InitFlags.NONE, None)
        
        if result == Error.Success:
            print("Logitech驱动初始化成功")
            is_initialized = True
            return True
        else:
            print(f"初始化失败，错误码: {result}")
            if result == Error.DeviceNotFound:
                print("找不到设备，请确保已安装Logitech G HUB或Logitech Gaming Software")
            elif result == Error.InvalidArgument:
                print("参数无效")
            else:
                print("未知错误")
            return False
                
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False

def cleanup_keyboard_simulator():
    """清理键盘模拟器资源"""
    global send_lib, is_initialized
    
    if is_initialized and send_lib:
        try:
            send_lib.IbAhkSendDestroy()
            print("\n键盘模拟器资源已清理")
            is_initialized = False
        except Exception as e:
            print(f"\n清理键盘模拟器资源时发生错误: {str(e)}")

def check_logitech_driver():
    """检查Logitech驱动是否可用"""
    global send_lib, is_initialized
    
    if not is_initialized:
        print("\nLogitech驱动未初始化，正在尝试初始化...")
        if not initialize_keyboard_simulator():
            print("\nLogitech驱动不可用，程序将退出")
            return False
    return True

def simulate_key_press(key, duration=0.05):
    """使用IbAhkSendLib模拟按键
    
    Args:
        key: 要模拟的按键，可以是字符串或虚拟键码
        duration: 按键持续时间，默认0.05秒
    """
    global send_lib, is_initialized
    
    # 检查驱动状态
    if not check_logitech_driver():
        return False
    
    try:
        # 获取虚拟键码
        if isinstance(key, str):
            if key.upper() == 'O':
                vk_code = VK_O
            elif key.upper() == 'I':
                vk_code = VK_I
            elif key.upper() == 'L':
                vk_code = VK_L
            else:
                vk_code = ord(key.upper())
        else:
            vk_code = key
        
        # 创建一个INPUT结构体用于按下键
        num_inputs = 1
        inputs = (Input * num_inputs)()
        
        # 设置为键盘输入
        inputs[0].type = INPUT_KEYBOARD
        
        # 按下键
        inputs[0].ii.ki.wVk = vk_code
        inputs[0].ii.ki.wScan = 0
        inputs[0].ii.ki.dwFlags = 0
        inputs[0].ii.ki.time = 0
        inputs[0].ii.ki.dwExtraInfo = ctypes.pointer(wintypes.ULONG(0))
        
        # 发送按下输入
        print(f"\n正在发送 '{key}' 键按下...")
        send_result = send_lib.IbAhkSendInput(num_inputs, inputs, ctypes.sizeof(Input))
        print(f"发送按下结果: {send_result}")
        
        # 等待指定的持续时间
        time.sleep(duration)
        
        # 创建一个INPUT结构体用于释放键
        inputs[0].ii.ki.dwFlags = KEYEVENTF_KEYUP
        
        # 发送释放输入
        print(f"正在发送 '{key}' 键释放...")
        send_result = send_lib.IbAhkSendInput(num_inputs, inputs, ctypes.sizeof(Input))
        print(f"发送释放结果: {send_result}")
        
        return True
    except Exception as e:
        print(f"\n按键模拟失败: {str(e)}")
        print("按键模拟失败，程序将退出")
        return False 