import cv2
import numpy as np
import pyautogui

def grab_screen(region=None):
    """
    使用pyautogui代替win32api获取屏幕截图
    
    Args:
        region: 截图区域，格式为(top, left, width, height)
        
    Returns:
        屏幕截图的numpy数组，格式为BGR
    """
    if region:
        top, left, width, height = region
        # pyautogui的区域格式为(left, top, width, height)
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
    else:
        screenshot = pyautogui.screenshot()
    
    # 转换为numpy数组并调整颜色通道顺序
    img = np.array(screenshot)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)