import torch
import cv2
import numpy as np
import time
import sys
import keyboard
import mss
import os
import random
import kmNet

from models.config import *
from models.load_model import get_model
from models.augmentations import letterbox
from models.general import check_img_size, non_max_suppression, scale_coords

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_one_box() input.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# 目标检测配置
TARGET_CLASS = 0  # 0: hit, 1: me - 检测hit类别
DETECTION_COOLDOWN = 0.05  # 50ms冷却时间

# 按键配置 - 统一检测逻辑，根据位置关系判断是否输出
KEY_CONFIGS = {
    'm': {
        'trigger_key': 'm',
        'output_key': None,  # M键不输出任何按键
        'max_no_detection_frames': 14,
        'description': 'M键触发位置检测（无输出）'
    },
    'k': {
        'trigger_key': 'k',
        'output_key': None,  # K键条件输出
        'max_no_detection_frames': 14,
        'description': 'K键触发位置检测（W键按下时无输出，W键未按下时输出U键）'
    },
    'l': {
        'trigger_key': 'l',
        'output_key': None,  # L键条件输出
        'max_no_detection_frames': 14,
        'description': 'L键触发位置检测（W键按下时输出N键，W键未按下时无输出）'
    },
    'p': {
        'trigger_key': 'p',
        'output_key': None,  # P键不输出
        'max_no_detection_frames': 14,
        'description': 'P键触发位置检测（无输出）'
    },
    'i': {
        'trigger_key': 'i',
        'output_key': None,  # I键条件输出
        'max_no_detection_frames': 14,  # I键延长至14帧
        'description': 'I键触发位置检测（W键按下时输出I+O键，W键未按下时输出N键）'
    },
    'o': {
        'trigger_key': 'o',
        'output_key': None,  # O键不输出
        'max_no_detection_frames': 14,  # O键延长至14帧
        'description': 'O键触发位置检测（无输出）'
    },
    'n': {
        'trigger_key': 'n',
        'output_key': None,  # N键条件输出
        'max_no_detection_frames': 14,
        'description': 'N键触发位置检测（W键按下时无输出，W键未按下时输出I+O键）'
    }
}

# 简化配置
CONFIG = {
    'model_weights': 'MyTrainFor_SF6.pt',
    'conf_threshold': 0.3,  # 降低置信度阈值
    'iou_threshold': 0.45,
    'classes': [0, 1, 2],  # 0,1,2 与特瑞.py保持一致，2为hit
    'device': '',
    'half': True,
    'max_det': 1000,
    'key_codes': {
        'w': 26,  # W键的键码 (第23个字母 + 3)
        'o': 18,  # O键的键码 (第15个字母 + 3)
        'm': 16,  # M键的键码 (第13个字母 + 3)
        'i': 12,  # I键的键码 (第9个字母 + 3)
        'l': 15,  # L键的键码 (第12个字母 + 3) - 锚点
        'p': 19,  # P键的键码 (第16个字母 + 3)
        'n': 17,  # N键的键码 (第14个字母 + 3)
        'k': 14,  # K键的键码 (第11个字母 + 3)
        'j': 13,  # J键的键码 (第10个字母 + 3)
        'u': 24,  # U键的键码 (第22个字母 + 3)
        'g': 10,  # G键的键码 (第7个字母 + 3)
        'space': 44,   # 空格键的键码
        'a': 4,   # A键的键码 (第1个字母 + 3)
        'd': 7,   # D键的键码 (第4个字母 + 3)
        'q': 20,  # Q键的键码 (第17个字母 + 3)
        'e': 8,   # E键的键码 (第5个字母 + 3)
        'comma': 54   # 逗号键的键码
    }
}

# 基于kmNet的物理按键屏蔽冷却配置
COOLDOWN_KEYS = ['k', 'l', 'm', 'n', 'o', 'i', 'space']
# 触发键对应的冷却时长（秒）
TRIGGER_COOLDOWN_S = {
    'k': 0.30,
    'l': 0.30,
    'm': 0.1,
    'n': 0.20,
    'o': 0.30,
    'i': 0.20,
}

# 截图区域（与特瑞.py一致的更大窗口，便于稳定检测）
MONITOR = {'top': 44, 'left': 0, 'width': 1920, 'height': 600}

class MultiKeyDetector:
    def __init__(self):
        self.frame_count = 0
        self.total_detection_count = 0
        self.system_active = False  # 系统总开关
        self.is_locked_in = False   # 新增：锁定模式状态
        
        # 新增：w键状态监控
        self.w_key_state = False  # 全局w键状态
        self.w_key_thread = None  # w键监控线程
        
        # 新增：用户设定的class1位置
        self.user_set_class1_side = None  # 'left' 或 'right'，由z键(左)或c键(右)设定
        
        # 新增：功能模式跟踪（删除2键模式，只保留3键模式）
        self.mode = '3'  # 只有3键模式
        
        # 按键输出冷却机制
        self.key_output_cooldown = {}  # 记录每个按键的输出冷却时间
        self.cooldown_duration = 0.3   # 冷却时间300ms
        

        
        # 为每个按键创建独立的状态跟踪
        self.key_states = {}
        for key_name, config in KEY_CONFIGS.items():
            self.key_states[key_name] = {
                'detection_active': False,
                'key_triggered': False,
                'no_detection_frames': 0,
                'detection_count': 0,
                'w_key_state': False
            }
            self.key_output_cooldown[key_name] = 0  # 初始化冷却时间
        
        # kmNet物理屏蔽冷却状态
        self.mask_cooldown_end_by_key = {k: 0.0 for k in COOLDOWN_KEYS}
        self.masked_keys = set()

        # 键盘事件句柄
        self.key_event_hooks = {}
        self.handlers_registered = False
        
        # 全局宏事件句柄（常驻，不随1/R改变）
        self.global_macro_hooks = {}
        
        # 添加win32api导入
        try:
            import win32api
            import win32con
            self.use_win32api = True
            print("成功加载win32api，将使用底层键盘监听")
        except ImportError:
            self.use_win32api = False
            print("未找到win32api，将使用keyboard库")
        
        # 延迟初始化资源，按1键后再启动
        self.kmbox_ready = False
        self.mss_instance = None
        self.model = None
        self.device = None
        self.half = None
        self.stride = None
        self.names = None
        self.w_key_monitor_running = False

        # 注册全局宏（始终在线）
        self.register_global_macro_handlers()
        
        # 快速路径重入保护
        self.fast_path_running = False
        
        # 按键事件计时状态
        self.key_event_timers = {}  # 记录每个按键事件的开始时间
        self.timing_enabled = True  # 计时开关
        self.global_macros_enabled = True  # 全局宏开关（V 键切换；关闭后 j/m/a/d/逗号 可正常打字）
        
        # 初始化按键计时器
        for key_name in KEY_CONFIGS.keys():
            self.key_event_timers[key_name] = None

        # 冷却期间被屏蔽按键的物理按下记录与监控
        import threading
        self.recorded_masked_key_presses = set()
        self.recorded_keys_lock = threading.Lock()
        self.cooldown_press_monitor_running = False
        self.cooldown_press_monitor_thread = None
        # 预/实屏蔽期间需要监听的目标键集合（包含space）
        self.cooldown_monitor_targets = set()
        # 冷却活动标志与预屏蔽截止时间（覆盖按下开始到冷却结束，包含0.05s延迟）
        self.cooldown_active = False
        self.cooldown_pre_mask_deadline = 0.0
        # 监听窗口起始时间与已记录日志的键（避免重复日志）
        self.cooldown_monitor_start_ts = 0.0
        self.monitor_logged_keys = set()
        
        # 智能宏状态管理
        self.smart_macro_states = {
            'a': {
                'original_macro_active': False,  # 原宏是否激活
                'temp_macro_active': False,      # 临时宏是否激活
                'temp_macro_end_time': 0.0,     # 临时宏结束时间
                'last_press_time': 0.0          # 上次按下时间
            },
            'd': {
                'original_macro_active': False,  # 原宏是否激活
                'temp_macro_active': False,      # 临时宏是否激活
                'temp_macro_end_time': 0.0,     # 临时宏结束时间
                'last_press_time': 0.0          # 上次按下时间
            },
            'space': {
                'original_macro_active': False,  # 原宏是否激活
                'temp_macro_active': False,      # 临时宏是否激活
                'temp_macro_end_time': 0.0,     # 临时宏结束时间
                'last_press_time': 0.0          # 上次按下时间
            }
        }

        # W键屏蔽保护机制（防止高频操作时W键持续输出）
        import threading
        self.w_mask_ref_count = 0  # W键屏蔽引用计数
        self.w_mask_lock = threading.Lock()  # W键屏蔽操作锁
        self.w_key_pressed_by_macro = False  # 标记宏是否按下W键
        # 物理w键监听机制（选项C：在屏蔽期间监听并忽略物理按键）
        self.w_physical_press_tracked = False  # 标记屏蔽期间是否检测到物理w键按下
        self.w_physical_monitor_thread = None  # 物理w键监听线程
        self.w_physical_monitor_running = False  # 监听线程运行标志
        self.w_physical_monitor_lock = threading.Lock()  # 物理监听状态锁

    def ensure_w_key_released(self):
        """确保W键已释放（防止W键持续输出）- 同时检查kmNet内部状态和物理按键状态"""
        try:
            w_hid = CONFIG['key_codes']['w']
            released_kmnet = False
            released_physical = False
            
            # 1. 检查kmNet内部状态
            try:
                is_pressed = kmNet.isdown_keyboard(w_hid)
                if is_pressed == 1:
                    # W键仍被按下，强制释放
                    kmNet.keyup(w_hid)
                    print("[W键保护] 检测到kmNet内部W键未释放，已强制释放")
                    time.sleep(0.01)  # 短暂延迟确保释放完成
                    released_kmnet = True
            except Exception as e:
                print(f"[W键保护] 检查kmNet W键状态失败: {e}")
                # 即使检查失败，也尝试释放一次
                try:
                    kmNet.keyup(w_hid)
                except:
                    pass
            
            # 2. 检查物理按键状态（如果之前有记录）
            with self.w_physical_monitor_lock:
                if self.w_physical_press_tracked:
                    # 检测物理w键是否仍被按下
                    try:
                        if self.use_win32api:
                            import win32api
                            VK_W = 0x57
                            physical_pressed = win32api.GetAsyncKeyState(VK_W) & 0x8000 != 0
                        else:
                            physical_pressed = keyboard.is_pressed('w')
                        
                        if physical_pressed:
                            print("[W键保护] 检测到物理W键仍被按下，等待松开...")
                            # 等待物理w键松开（最多等待200ms）
                            max_wait = 0.2
                            wait_start = time.perf_counter()
                            while physical_pressed and (time.perf_counter() - wait_start) < max_wait:
                                time.sleep(0.01)
                                if self.use_win32api:
                                    physical_pressed = win32api.GetAsyncKeyState(VK_W) & 0x8000 != 0
                                else:
                                    physical_pressed = keyboard.is_pressed('w')
                            
                            if physical_pressed:
                                print("[W键保护] 警告：物理W键在等待后仍被按下，强制清除记录")
                            else:
                                print("[W键保护] 物理W键已松开")
                            
                            released_physical = True
                        else:
                            print("[W键保护] 物理W键已松开（记录清除）")
                            released_physical = True
                        
                        # 清除记录
                        self.w_physical_press_tracked = False
                    except Exception as e:
                        print(f"[W键保护] 检查物理W键状态失败: {e}")
                        # 清除记录以防万一
                        self.w_physical_press_tracked = False
            
            return released_kmnet or released_physical
        except Exception as e:
            print(f"[W键保护] 确保W键释放失败: {e}")
            return False

    def safe_mask_w_key(self):
        """安全屏蔽W键（带引用计数，防止并发冲突）"""
        with self.w_mask_lock:
            self.w_mask_ref_count += 1
            if self.w_mask_ref_count == 1:
                # 第一次屏蔽，执行实际屏蔽操作
                try:
                    kmNet.mask_keyboard(CONFIG['key_codes']['w'])
                    print(f"[W键保护] 屏蔽W键（引用计数: {self.w_mask_ref_count}）")
                    # 启动物理w键监听线程（选项C：在屏蔽期间监听并忽略物理按键）
                    self.start_w_physical_monitor()
                except Exception as e:
                    print(f"[W键保护] 屏蔽W键失败: {e}")
                    self.w_mask_ref_count -= 1  # 失败时回退计数
            else:
                print(f"[W键保护] W键已屏蔽，增加引用计数: {self.w_mask_ref_count}")

    def safe_unmask_w_key(self):
        """安全解除W键屏蔽（带引用计数，解除前强制释放W键）"""
        with self.w_mask_lock:
            if self.w_mask_ref_count <= 0:
                print("[W键保护] 警告：引用计数异常，已为0或负数")
                return
            
            # 停止物理w键监听线程（选项C：在解除屏蔽前停止监听）
            self.stop_w_physical_monitor()
            
            # 解除屏蔽前，强制确保W键已释放（包括物理按键状态）
            self.ensure_w_key_released()
            
            self.w_mask_ref_count -= 1
            if self.w_mask_ref_count == 0:
                # 引用计数归零，执行实际解除屏蔽操作
                try:
                    kmNet.unmask_keyboard(CONFIG['key_codes']['w'])
                    print(f"[W键保护] 解除W键屏蔽（引用计数: {self.w_mask_ref_count}）")
                except Exception as e:
                    print(f"[W键保护] 解除W键屏蔽失败: {e}")
            else:
                print(f"[W键保护] W键仍被其他宏屏蔽，减少引用计数: {self.w_mask_ref_count}")

    def start_w_physical_monitor(self):
        """启动物理w键监听线程（选项C：在屏蔽期间监听并忽略物理按键）"""
        with self.w_physical_monitor_lock:
            if self.w_physical_monitor_running:
                return  # 已经在运行
            
            self.w_physical_monitor_running = True
            self.w_physical_press_tracked = False
            
            def monitor_w_physical():
                """监听物理w键按下/松开，仅记录状态不处理"""
                last_state = False
                try:
                    if self.use_win32api:
                        import win32api
                        VK_W = 0x57
                    else:
                        VK_W = None
                    
                    while self.w_physical_monitor_running:
                        try:
                            # 检测物理w键状态
                            if self.use_win32api and VK_W:
                                current_state = win32api.GetAsyncKeyState(VK_W) & 0x8000 != 0
                            else:
                                current_state = keyboard.is_pressed('w')
                            
                            # 检测到状态变化
                            if current_state != last_state:
                                with self.w_physical_monitor_lock:
                                    if current_state:
                                        # 物理w键被按下（在屏蔽期间）
                                        self.w_physical_press_tracked = True
                                        print(f"[W键物理监听] 检测到物理W键按下（屏蔽期间，仅记录不处理）")
                                    else:
                                        # 物理w键松开
                                        if self.w_physical_press_tracked:
                                            print(f"[W键物理监听] 检测到物理W键松开（清除记录）")
                                            self.w_physical_press_tracked = False
                                
                                last_state = current_state
                            
                            time.sleep(0.01)  # 10ms轮询间隔
                        except Exception as e:
                            print(f"[W键物理监听] 检测错误: {e}")
                            time.sleep(0.01)
                except Exception as e:
                    print(f"[W键物理监听] 线程错误: {e}")
                finally:
                    with self.w_physical_monitor_lock:
                        self.w_physical_monitor_running = False
                    print("[W键物理监听] 监听线程已停止")
            
            import threading
            self.w_physical_monitor_thread = threading.Thread(target=monitor_w_physical, daemon=True)
            self.w_physical_monitor_thread.start()
            print("[W键物理监听] 监听线程已启动")

    def stop_w_physical_monitor(self):
        """停止物理w键监听线程"""
        with self.w_physical_monitor_lock:
            if not self.w_physical_monitor_running:
                return  # 未在运行
            
            self.w_physical_monitor_running = False
        
        # 等待线程结束（最多等待100ms）
        if self.w_physical_monitor_thread and self.w_physical_monitor_thread.is_alive():
            self.w_physical_monitor_thread.join(timeout=0.1)
            if self.w_physical_monitor_thread.is_alive():
                print("[W键物理监听] 警告：监听线程未在预期时间内停止")
            else:
                print("[W键物理监听] 监听线程已停止")

    def ensure_kmbox_ready(self):
        if not self.kmbox_ready:
            print("[宏] 初始化kmbox连接...")
            if not self.initialize_kmbox():
                print("[宏] kmbox初始化失败")
                return False
            self.kmbox_ready = True
            print("[宏] kmbox初始化成功")
        return True

    def on_global_j_press(self):
        """J键全局宏：W按下保持20~40ms，松开W的同时点击一次K。"""
        if not self.global_macros_enabled:
            return
        try:
            if not self.ensure_kmbox_ready():
                return
            import threading
            def worker():
                try:
                    dur = random.randint(15, 20) / 1000
                    # W按下保持
                    kmNet.keydown(CONFIG['key_codes']['w'])
                    time.sleep(dur)
                    # 松开W并立即点击K
                    kmNet.keyup(CONFIG['key_codes']['w'])
                    # K点击
                    durk = random.randint(40, 60) / 1000
                    kmNet.keydown(CONFIG['key_codes']['m'])
                    time.sleep(durk)
                    kmNet.keyup(CONFIG['key_codes']['m'])
                    print(f"[宏J] W保持{dur*1000:.0f}ms，随后点击K完成")
                except Exception as e:
                    print(f"[宏J] 执行失败: {e}")
            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            print(f"[宏J] 触发失败: {e}")

    def on_global_m_press(self):
        """M键全局宏：不拦截M，只额外触发一次W保持30~60ms。"""
        if not self.global_macros_enabled:
            return
        try:
            if not self.ensure_kmbox_ready():
                return
            import threading
            def worker():
                try:
                    dur = random.randint(30, 60) / 1000

                    print(f"[宏M] 额外W保持{dur*1000:.0f}ms完成")
                except Exception as e:
                    print(f"[宏M] 执行失败: {e}")
            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            print(f"[宏M] 触发失败: {e}")

    def on_global_a_press(self):
        """A键智能宏：原宏→临时宏→临时宏输出"""
        if not self.global_macros_enabled:
            return
        try:
            if not self.ensure_kmbox_ready():
                return
            import threading
            def worker():
                try:
                    current_time = time.perf_counter()
                    macro_state = self.smart_macro_states['a']
                    
                    # 检查是否在临时宏期间
                    if macro_state['temp_macro_active'] and current_time < macro_state['temp_macro_end_time']:
                        # 临时宏期间：执行临时宏输出
                        print("[宏A] 临时宏期间，执行临时宏输出")
                        self.execute_temp_macro_a()
                    else:
                        # 执行原宏
                        print("[宏A] 执行原宏")
                        self.execute_original_macro_a()
                        # 启动临时宏替换
                        self.start_temp_macro_a()
                        
                except Exception as e:
                    print(f"[宏A] 执行失败: {e}")
            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            print(f"[宏A] 触发失败: {e}")

    def on_global_d_press(self):
        """D键智能宏：原宏→临时宏→临时宏输出"""
        if not self.global_macros_enabled:
            return
        try:
            if not self.ensure_kmbox_ready():
                return
            import threading
            def worker():
                try:
                    current_time = time.perf_counter()
                    macro_state = self.smart_macro_states['d']
                    
                    # 检查是否在临时宏期间
                    if macro_state['temp_macro_active'] and current_time < macro_state['temp_macro_end_time']:
                        # 临时宏期间：执行临时宏输出
                        print("[宏D] 临时宏期间，执行临时宏输出")
                        self.execute_temp_macro_d()
                    else:
                        # 执行原宏
                        print("[宏D] 执行原宏")
                        self.execute_original_macro_d()
                        # 启动临时宏替换
                        self.start_temp_macro_d()
                        
                except Exception as e:
                    print(f"[宏D] 执行失败: {e}")
            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            print(f"[宏D] 触发失败: {e}")

    def on_global_space_press(self):
        """Space键智能宏：原宏→临时宏→临时宏输出（仅按W键，不输出其他按键）"""
        try:
            if not self.ensure_kmbox_ready():
                return
            import threading
            def worker():
                try:
                    current_time = time.perf_counter()
                    macro_state = self.smart_macro_states['space']
                    
                    # 检查是否在临时宏期间
                    if macro_state['temp_macro_active'] and current_time < macro_state['temp_macro_end_time']:
                        # 临时宏期间：执行临时宏输出
                        print("[宏Space] 临时宏期间，执行临时宏输出")
                        self.execute_temp_macro_space()
                    else:
                        # 执行原宏
                        print("[宏Space] 执行原宏")
                        self.execute_original_macro_space()
                        # 启动临时宏替换
                        self.start_temp_macro_space()
                        
                except Exception as e:
                    print(f"[宏Space] 执行失败: {e}")
            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            print(f"[宏Space] 触发失败: {e}")

    def on_global_comma_press(self):
        """逗号键全局宏：复杂按键序列，松开逗号键可停止"""
        if not self.global_macros_enabled:
            return
        try:
            if not self.ensure_kmbox_ready():
                return
            import threading
            
            # 创建停止标志（使用简单的布尔值，避免复杂检查）
            self.comma_macro_stop_flag = False
            self.comma_macro_thread = None
            
            def worker():
                try:
                    # 快速检查停止标志（只在关键节点检查，减少开销）
                    if self.comma_macro_stop_flag:
                        return
                    
                    kmNet.keydown(CONFIG['key_codes']['k'])
                    kmNet.keydown(CONFIG['key_codes']['l'])
                    time.sleep(60 / 1000)
                    kmNet.keyup(CONFIG['key_codes']['k'])
                    kmNet.keyup(CONFIG['key_codes']['l'])
                    
                    # 只在长时间延迟前检查
                    if self.comma_macro_stop_flag:
                        return
                    
                    time.sleep(970/1000)
                    
                    if self.comma_macro_stop_flag:
                        return
                    
                     # W (30~60ms)
                    kmNet.keydown(CONFIG['key_codes']['w'])
                    time.sleep(45 / 1000)
                    kmNet.keydown(CONFIG['key_codes']['q'])
                    time.sleep(45 / 1000)            
                    kmNet.keyup(CONFIG['key_codes']['w'])
                    time.sleep(45 / 1000)  
                    kmNet.keyup(CONFIG['key_codes']['q'])
                    
                    # 只在长时间延迟前检查
                    if self.comma_macro_stop_flag:
                        return
                    
                     # W (30~60ms)
                    kmNet.keydown(CONFIG['key_codes']['w'])
                    time.sleep(45 / 1000)
                    kmNet.keydown(CONFIG['key_codes']['q'])
                    time.sleep(45 / 1000)            
                    kmNet.keyup(CONFIG['key_codes']['w'])
                    time.sleep(45 / 1000)  
                    kmNet.keyup(CONFIG['key_codes']['q'])

                    kmNet.keydown(CONFIG['key_codes']['k'])
                    time.sleep(60 / 1000)
                    kmNet.keyup(CONFIG['key_codes']['k'])

                    time.sleep(2270/1000)
                    
                    if self.comma_macro_stop_flag:
                        return
                    
                     # W (30~60ms)
                    kmNet.keydown(CONFIG['key_codes']['w'])
                    time.sleep(45 / 1000)
                    kmNet.keydown(CONFIG['key_codes']['q'])
                    time.sleep(45 / 1000)            
                    kmNet.keyup(CONFIG['key_codes']['w'])
                    time.sleep(45 / 1000)  
                    kmNet.keyup(CONFIG['key_codes']['q'])

                    time.sleep(45 / 1000)
                    kmNet.keydown(CONFIG['key_codes']['l'])
                    time.sleep(60 / 1000)
                    kmNet.keyup(CONFIG['key_codes']['l'])

                    print(f"[宏逗号] 复杂按键序列完成")
                except Exception as e:
                    print(f"[宏逗号] 执行失败: {e}")
                finally:
                    # 快速清理（避免复杂检查）
                    self.comma_macro_stop_flag = False
                    self.comma_macro_thread = None
            
            # 启动宏线程
            self.comma_macro_thread = threading.Thread(target=worker, daemon=True)
            self.comma_macro_thread.start()
            print("[宏逗号] 复杂按键序列开始执行")
        except Exception as e:
            print(f"[宏逗号] 触发失败: {e}")

    def on_global_comma_release(self):
        """逗号键松开：停止正在执行的宏（轻量级处理）"""
        try:
            # 直接设置标志，避免复杂检查
            self.comma_macro_stop_flag = True
        except Exception:
            pass  # 静默处理，不影响其他功能

    def execute_original_macro_a(self):
        """执行A键原宏：Q 20ms q W (30~60ms) Q (30~60ms) w (30~60ms) q"""
        w_key_pressed = False
        try:
            # A键立即屏蔽W键（使用安全屏蔽方法）
            self.safe_mask_w_key()
            
            # Q 20ms q
            kmNet.keydown(CONFIG['key_codes']['q'])
            time.sleep(0.02)
            kmNet.keyup(CONFIG['key_codes']['q'])
            
            # W (30~60ms)
            w_dur = random.randint(20, 45) / 1000
            kmNet.keydown(CONFIG['key_codes']['w'])
            w_key_pressed = True
            time.sleep(w_dur)

            kmNet.keydown(CONFIG['key_codes']['q'])

            q_dur = random.randint(20, 45) / 1000
            time.sleep(q_dur)            
            kmNet.keyup(CONFIG['key_codes']['w'])
            w_key_pressed = False
            
            time.sleep(q_dur)  
            kmNet.keyup(CONFIG['key_codes']['q'])
            
            # w 结束
            print(f"[宏A原宏] 完成：Q20ms q W{w_dur*1000:.0f}ms Q{q_dur*1000:.0f}ms q")
        except Exception as e:
            print(f"[宏A原宏] 执行失败: {e}")
        finally:
            # 确保W键已释放
            if w_key_pressed:
                try:
                    kmNet.keyup(CONFIG['key_codes']['w'])
                except:
                    pass
            # 安全解除W键屏蔽（会自动检查并释放W键）
            self.safe_unmask_w_key()

    def execute_temp_macro_a(self):
        """执行A键临时宏：W (30~60ms) Q (30~60ms) w (30~60ms) q"""
        w_key_pressed = False
        try:
            # A键临时宏立即屏蔽W键（使用安全屏蔽方法）
            self.safe_mask_w_key()
            
            # W (30~60ms)
            w_dur = random.randint(20, 30) / 1000
            kmNet.keydown(CONFIG['key_codes']['w'])
            w_key_pressed = True
            time.sleep(w_dur)
            # Q (30~60ms)
            q_dur = random.randint(20, 30) / 1000
            kmNet.keydown(CONFIG['key_codes']['q'])
            time.sleep(q_dur)
            kmNet.keyup(CONFIG['key_codes']['w'])
            w_key_pressed = False
            
            time.sleep(q_dur)
            kmNet.keyup(CONFIG['key_codes']['q'])
            
            print(f"[宏A临时宏] 完成：W{w_dur*1000:.0f}ms Q{q_dur*1000:.0f}ms q")
        except Exception as e:
            print(f"[宏A临时宏] 执行失败: {e}")
        finally:
            # 确保W键已释放
            if w_key_pressed:
                try:
                    kmNet.keyup(CONFIG['key_codes']['w'])
                except:
                    pass
            # 安全解除W键屏蔽（会自动检查并释放W键）
            self.safe_unmask_w_key()

    def execute_original_macro_d(self):
        """执行D键原宏：E 20ms e W (30~60ms) E (30~60ms) w (30~60ms) e"""
        w_key_pressed = False
        try:
            # D键立即屏蔽W键（使用安全屏蔽方法）
            self.safe_mask_w_key()
            
            # E 20ms e
            kmNet.keydown(CONFIG['key_codes']['e'])
            time.sleep(0.02)
            kmNet.keyup(CONFIG['key_codes']['e'])
            
            # W (30~60ms)
            w_dur = random.randint(20, 45) / 1000
            kmNet.keydown(CONFIG['key_codes']['w'])
            w_key_pressed = True
            time.sleep(w_dur)

            
            # E (30~60ms)
            e_dur = random.randint(14, 45) / 1000
            kmNet.keydown(CONFIG['key_codes']['e'])
            time.sleep(e_dur)
            kmNet.keyup(CONFIG['key_codes']['w'])
            w_key_pressed = False
            
            time.sleep(e_dur)            
            kmNet.keyup(CONFIG['key_codes']['e'])
            
            print(f"[宏D原宏] 完成：E20ms e W{w_dur*1000:.0f}ms E{e_dur*1000:.0f}ms e")
        except Exception as e:
            print(f"[宏D原宏] 执行失败: {e}")
        finally:
            # 确保W键已释放
            if w_key_pressed:
                try:
                    kmNet.keyup(CONFIG['key_codes']['w'])
                except:
                    pass
            # 安全解除W键屏蔽（会自动检查并释放W键）
            self.safe_unmask_w_key()

    def execute_original_macro_space(self):
        """执行Space键原宏：仅按W键 (20~45ms)，不输出其他按键"""
        w_key_pressed = False
        try:
            # Space键立即屏蔽W键和Space键（使用安全屏蔽方法）
            self.safe_mask_w_key()
            self.safe_mask_space_key()
            
            # W (20~45ms)
            w_dur = random.randint(20, 45) / 1000
            kmNet.keydown(CONFIG['key_codes']['w'])
            w_key_pressed = True
            time.sleep(w_dur)
            kmNet.keyup(CONFIG['key_codes']['w'])
            w_key_pressed = False
            
            print(f"[宏Space原宏] 完成：W{w_dur*1000:.0f}ms")
        except Exception as e:
            print(f"[宏Space原宏] 执行失败: {e}")
        finally:
            # 确保W键已释放
            if w_key_pressed:
                try:
                    kmNet.keyup(CONFIG['key_codes']['w'])
                except:
                    pass
            # 安全解除W键和Space键屏蔽（会自动检查并释放W键和Space键）
            self.safe_unmask_w_key()
            self.safe_unmask_space_key()

    def execute_temp_macro_d(self):
        """执行D键临时宏：W (30~60ms) E (30~60ms) w (30~60ms) e"""
        w_key_pressed = False
        try:
            # D键临时宏立即屏蔽W键（使用安全屏蔽方法）
            self.safe_mask_w_key()
            
            # W (30~60ms)
            w_dur = random.randint(14, 30) / 1000
            kmNet.keydown(CONFIG['key_codes']['w'])
            w_key_pressed = True
            time.sleep(w_dur)

            
            # E (30~60ms)
            e_dur = random.randint(15, 30) / 1000
            kmNet.keydown(CONFIG['key_codes']['e'])
            time.sleep(e_dur)
            kmNet.keyup(CONFIG['key_codes']['w'])
            w_key_pressed = False
            
            time.sleep(e_dur)            
            kmNet.keyup(CONFIG['key_codes']['e'])
            
            print(f"[宏D临时宏] 完成：W{w_dur*1000:.0f}ms E{e_dur*1000:.0f}ms e")
        except Exception as e:
            print(f"[宏D临时宏] 执行失败: {e}")
        finally:
            # 确保W键已释放
            if w_key_pressed:
                try:
                    kmNet.keyup(CONFIG['key_codes']['w'])
                except:
                    pass
            # 安全解除W键屏蔽（会自动检查并释放W键）
            self.safe_unmask_w_key()

    def execute_temp_macro_space(self):
        """执行Space键临时宏：仅按W键 (14~30ms)，不输出其他按键"""
        w_key_pressed = False
        try:
            # Space键临时宏立即屏蔽W键和Space键（使用安全屏蔽方法）
            self.safe_mask_w_key()
            self.safe_mask_space_key()
            
            # W (14~30ms)
            w_dur = random.randint(14, 30) / 1000
            kmNet.keydown(CONFIG['key_codes']['w'])
            w_key_pressed = True
            time.sleep(w_dur)
            kmNet.keyup(CONFIG['key_codes']['w'])
            w_key_pressed = False
            
            print(f"[宏Space临时宏] 完成：W{w_dur*1000:.0f}ms")
        except Exception as e:
            print(f"[宏Space临时宏] 执行失败: {e}")
        finally:
            # 确保W键已释放
            if w_key_pressed:
                try:
                    kmNet.keyup(CONFIG['key_codes']['w'])
                except:
                    pass
            # 安全解除W键和Space键屏蔽（会自动检查并释放W键和Space键）
            self.safe_unmask_w_key()
            self.safe_unmask_space_key()

    def start_temp_macro_a(self):
        """启动A键临时宏（200ms有效期）"""
        macro_state = self.smart_macro_states['a']
        macro_state['temp_macro_active'] = True
        macro_state['temp_macro_end_time'] = time.perf_counter() + 0.2  # 200ms
        print("[宏A] 临时宏已激活，有效期200ms")

    def start_temp_macro_d(self):
        """启动D键临时宏（200ms有效期）"""
        macro_state = self.smart_macro_states['d']
        macro_state['temp_macro_active'] = True
        macro_state['temp_macro_end_time'] = time.perf_counter() + 0.2  # 200ms
        print("[宏D] 临时宏已激活，有效期200ms")

    def start_temp_macro_space(self):
        """启动Space键临时宏（200ms有效期）"""
        macro_state = self.smart_macro_states['space']
        macro_state['temp_macro_active'] = True
        macro_state['temp_macro_end_time'] = time.perf_counter() + 0.2  # 200ms
        print("[宏Space] 临时宏已激活，有效期200ms")

    def cleanup_expired_temp_macros(self):
        """清理过期的临时宏状态"""
        current_time = time.perf_counter()
        for key, macro_state in self.smart_macro_states.items():
            if (macro_state['temp_macro_active'] and 
                current_time >= macro_state['temp_macro_end_time']):
                macro_state['temp_macro_active'] = False
                macro_state['temp_macro_end_time'] = 0.0
                if self.frame_count % 60 == 0:  # 每60帧显示一次清理状态
                    print(f"[宏{key.upper()}] 临时宏已过期，自动清理")

    def register_global_macro_handlers(self):
        try:
            if 'j' not in self.global_macro_hooks:
                self.global_macro_hooks['j'] = keyboard.on_press_key('j', lambda e: self.on_global_j_press())
            if 'm' not in self.global_macro_hooks:
                self.global_macro_hooks['m'] = keyboard.on_press_key('m', lambda e: self.on_global_m_press())
            if 'a' not in self.global_macro_hooks:
                self.global_macro_hooks['a'] = keyboard.on_press_key('a', lambda e: self.on_global_a_press())
            if 'd' not in self.global_macro_hooks:
                self.global_macro_hooks['d'] = keyboard.on_press_key('d', lambda e: self.on_global_d_press())
            if 'space' not in self.global_macro_hooks:
                self.global_macro_hooks['space'] = keyboard.on_press_key('space', lambda e: self.on_global_space_press())
            if ',' not in self.global_macro_hooks:
                self.global_macro_hooks[','] = keyboard.on_press_key(',', lambda e: self.on_global_comma_press())
                # 同时注册逗号键的松开事件
                keyboard.on_release_key(',', lambda e: self.on_global_comma_release())
            print("全局宏已注册：j(连W后K), m(附加W), a(智能宏), d(智能宏), space(智能宏-仅W), 逗号(复杂序列+松开停止)")
        except Exception as e:
            print(f"注册全局宏失败: {e}")

    def start_key_timer(self, key_name):
        """开始按键事件计时"""
        if self.timing_enabled:
            self.key_event_timers[key_name] = time.perf_counter()
            print(f"⏱️  {key_name.upper()}键计时开始")

    def get_elapsed_time(self, key_name):
        """获取按键事件已消耗的时间（毫秒）"""
        if not self.timing_enabled or self.key_event_timers.get(key_name) is None:
            return 0.0
        elapsed = (time.perf_counter() - self.key_event_timers[key_name]) * 1000
        return elapsed

    def log_timing(self, key_name, event_name):
        """记录并输出时间消耗"""
        if self.timing_enabled:
            elapsed = self.get_elapsed_time(key_name)
            print(f"⏱️  {key_name.upper()}键 {event_name}: {elapsed:.1f}ms")

    def stop_key_timer(self, key_name):
        """停止按键事件计时"""
        if self.timing_enabled and self.key_event_timers.get(key_name) is not None:
            total_time = self.get_elapsed_time(key_name)
            print(f"⏱️  {key_name.upper()}键总耗时: {total_time:.1f}ms")
            self.key_event_timers[key_name] = None

    def on_trigger_key_press(self, key_name):
        """基于事件的触发：处理本地键盘按下 k/l/m/n/o/i。"""
        try:
            if not self.system_active:
                return
            if self.is_locked_in:
                print(f"[锁定] 忽略按键按下: {key_name}")
                return
            if key_name not in KEY_CONFIGS:
                return
            key_config = KEY_CONFIGS[key_name]
            key_state = self.key_states[key_name]



            if not key_state['detection_active']:
                # 启动按键事件计时
                self.start_key_timer(key_name)
                
                # 立即记录W键状态
                if key_name in ['m', 'k', 'o', 'l', 'i', 'n']:
                    try:
                        import win32api
                        VK_W = 0x57
                        w_state = win32api.GetAsyncKeyState(VK_W) & 0x8000 != 0
                        key_state['w_key_state'] = w_state
                        print(f"\n[{key_name.upper()}键事件] 记录W键状态: {'按下' if w_state else '松开'} - {time.perf_counter():.3f}")
                    except Exception as e:
                        w_state = keyboard.is_pressed('w')
                        key_state['w_key_state'] = w_state
                        print(f"[DEBUG] 事件keyboard库检测W键: {w_state}")

                key_state['detection_active'] = True
                key_state['key_triggered'] = False
                key_state['no_detection_frames'] = 0

                # 重置hit坐标日志标志，允许新事件重新输出坐标
                if hasattr(self, '_hit_coords_logged'):
                    delattr(self, '_hit_coords_logged')
                print(f"\n*** {key_name.upper()}键事件触发！开始位置关系检测 ***")
                # 触发快速路径：立刻执行一次截图+检测，成功则立刻终止冷却
                try:
                    import threading
                    threading.Thread(target=self.fast_path_detect, args=(key_name,), daemon=True).start()
                except Exception as e:
                    print(f"[FastPath] 启动失败: {e}")
        except Exception as e:
            print(f"事件触发处理失败: {e}")

    def fast_path_detect(self, key_name):
        """快速路径：在按键事件中立刻执行一次YOLO检测，若满足条件则立即终止冷却并进入锁定模式。"""
        if self.fast_path_running:
            return
        self.fast_path_running = True
        try:
            if not self.system_active or self.model is None or self.mss_instance is None:
                return
            # 截图与检测
            img_orig = self.capture_screenshot()
            img_tensor = self.preprocess_image(img_orig)
            detections = self.detect_objects(img_tensor)
            if detections is not None and len(detections):
                detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], img_orig.shape).round()

            # 位置分析与判定
            class0_positions, class1_positions, _ = self.analyze_positions(detections)
            should_trigger, reason = self.should_trigger_output(class0_positions, class1_positions)
            if should_trigger and class0_positions:
                # 记录快速路径检测成功时间
                self.log_timing(key_name, "快速路径检测成功")
                
                best_class0 = max(class0_positions, key=lambda x: x['confidence'])
                self.send_key_output(key_name, best_class0['x_center'], best_class0['confidence'], reason)
                # 与常规路径一致：进入锁定模式并重置键状态
                print(f">>> {key_name.upper()}键触发（快速路径），进入锁定模式！等待Class 0消失...")
                self.is_locked_in = True
                for k in self.key_states:
                    self.key_states[k]['detection_active'] = False
                    self.key_states[k]['key_triggered'] = False
                    self.key_states[k]['no_detection_frames'] = 0
        except Exception as e:
            print(f"[FastPath] 错误: {e}")
        finally:
            self.fast_path_running = False

    def register_key_event_handlers(self):
        if self.handlers_registered:
            return
        try:
            for k in ['k', 'l', 'm', 'n', 'o', 'i', 'p']:
                if k in self.key_event_hooks:
                    continue
                # 同时注册按下与松开事件：按下仅进入检测流程，松开作为冷却监听开始点
                self.key_event_hooks[k] = (
                    keyboard.on_press_key(k, lambda e, key=k: self.on_trigger_key_press(key)),
                    keyboard.on_release_key(k, lambda e, key=k: self.on_trigger_key_release(key))
                )
            self.handlers_registered = True
            print("按键事件处理已注册：k,l,m,n,o,i,p")
        except Exception as e:
            print(f"注册按键事件失败: {e}")

    def unregister_key_event_handlers(self):
        if not self.handlers_registered:
            return
        try:
            for k, hook in list(self.key_event_hooks.items()):
                try:
                    # 解除按下与松开两个钩子
                    if isinstance(hook, tuple) and len(hook) == 2:
                        keyboard.unhook(hook[0])
                        keyboard.unhook(hook[1])
                    else:
                        keyboard.unhook(hook)
                except Exception:
                    pass
                self.key_event_hooks.pop(k, None)
            self.handlers_registered = False
            print("按键事件处理已注销")
        except Exception as e:
            print(f"注销按键事件失败: {e}")

    def on_trigger_key_release(self, key_name):
        """在触发键松开时开始冷却监听窗口并安排物理屏蔽（若需要）。"""
        try:
            if not self.system_active:
                return
            if self.is_locked_in:
                print(f"[锁定] 忽略按键松开: {key_name}")
                return
            if key_name not in KEY_CONFIGS:
                return
            # 仅对配置了冷却的触发键开启监听窗口
            if key_name in TRIGGER_COOLDOWN_S:
                # 监听窗口从松开时刻开始（覆盖0.05s预屏蔽）
                self.begin_cooldown_monitor_window(key_name)
                try:
                    import threading
                    def delayed_apply():
                        try:
                            time.sleep(0.01)
                            self.apply_cooldown_for_trigger(key_name)
                        except Exception as e:
                            print(f"延迟应用冷却失败: {e}")
                    threading.Thread(target=delayed_apply, daemon=True).start()
                except Exception as e:
                    print(f"启动延迟冷却线程失败: {e}")
        except Exception as e:
            print(f"释放事件处理失败: {e}")

    def apply_cooldown_for_trigger(self, trigger_key):
        """当系统未锁定时，屏蔽COOLDOWN_KEYS中所有按键一段时间。"""
        # 防止 stop_system 后延迟线程仍重新屏蔽按键
        if not self.system_active:
            return
        if self.is_locked_in:
            print(f"[锁定] 忽略冷却应用: {trigger_key}")
            return
        if trigger_key not in TRIGGER_COOLDOWN_S:
            return
        # 例外规则：
        # - k 屏蔽 n
        # - o 屏蔽 n, k
        # - i 屏蔽 k, n
        # - l 屏蔽 k, n
        # - n/m不屏蔽任何
        if trigger_key == 'l':
            # l屏蔽k,n
            target_keys = ['k', 'n']
        elif trigger_key == 'k':
            # k屏蔽n
            target_keys = ['n']
        elif trigger_key == 'o':
            # o屏蔽n,k
            target_keys = ['n', 'k']
        elif trigger_key == 'i':
            # i屏蔽k,n
            target_keys = ['n']
        elif trigger_key in ('n', 'm'):
            # n/m不屏蔽任何
            return
        else:
            target_keys = COOLDOWN_KEYS
        duration = TRIGGER_COOLDOWN_S[trigger_key]
        now = time.perf_counter()
        for key in target_keys:
            try:
                hid = CONFIG['key_codes'][key]
            except KeyError:
                continue
            # 屏蔽按键（若尚未屏蔽）
            if key not in self.masked_keys:
                try:
                    kmNet.mask_keyboard(hid)
                    self.masked_keys.add(key)
                except Exception as e:
                    print(f"屏蔽按键{key}失败: {e}")
            # 更新该键的冷却结束时间（取更晚的时间）
            end_time = now + duration
            if end_time > self.mask_cooldown_end_by_key.get(key, 0):
                self.mask_cooldown_end_by_key[key] = end_time

        # 从冷却开始刷新监听目标并启动监控（覆盖期已在按键按下时开启）
        self.refresh_cooldown_monitor_targets(target_keys)
        self.start_cooldown_press_monitor()

    def update_cooldowns(self):
        """在主循环中调用：对到期的冷却键解除屏蔽。"""
        now = time.perf_counter()
        for key in list(self.masked_keys):
            end_time = self.mask_cooldown_end_by_key.get(key, 0)
            if end_time and now >= end_time:
                try:
                    hid = CONFIG['key_codes'][key]
                    kmNet.unmask_keyboard(hid)
                except Exception as e:
                    print(f"解除屏蔽按键{key}失败: {e}")
                finally:
                    self.masked_keys.discard(key)
                    self.mask_cooldown_end_by_key[key] = 0.0

        # 若没有任何屏蔽键，停止冷却期按键监控线程
        if not self.masked_keys and getattr(self, 'cooldown_press_monitor_running', False):
            self.stop_cooldown_press_monitor()
            self.cooldown_active = False
        
        # 清理过期的临时宏状态
        self.cleanup_expired_temp_macros()

    def cancel_all_cooldowns_and_unmask(self):
        """立刻终止所有按键冷却并解除屏蔽。"""
        for key in list(self.masked_keys):
            try:
                hid = CONFIG['key_codes'][key]
                kmNet.unmask_keyboard(hid)
            except Exception as e:
                print(f"解除屏蔽按键{key}失败: {e}")
        # 清空状态
        self.masked_keys.clear()
        for key in COOLDOWN_KEYS:
            self.mask_cooldown_end_by_key[key] = 0.0

        # 停止冷却期按键监控
        if getattr(self, 'cooldown_press_monitor_running', False):
            self.stop_cooldown_press_monitor()
        self.cooldown_active = False

    def start_cooldown_press_monitor(self):
        """启动冷却期间被屏蔽按键的物理按下监控（包含space）。"""
        try:
            running = self.cooldown_press_monitor_running
        except AttributeError:
            running = False
        if running:
            return
        self.cooldown_press_monitor_running = True
        print("[监控] 冷却期按键监控启动")

        import threading
        def worker():
            while self.cooldown_press_monitor_running:
                try:
                    # 监听集合 = 实际被屏蔽的键 ∪ 预屏蔽覆盖目标
                    targets = set(self.masked_keys)
                    try:
                        targets |= set(self.cooldown_monitor_targets)
                    except Exception:
                        pass
                    if not targets:
                        time.sleep(0.002)
                        continue
                    # 遍历当前被屏蔽的键，记录实际按下
                    for key in list(targets):
                        try:
                            hid = CONFIG['key_codes'][key]
                        except KeyError:
                            continue
                        try:
                            state = kmNet.isdown_keyboard(hid)
                        except Exception:
                            state = 0
                        if state == 1:
                            try:
                                with self.recorded_keys_lock:
                                    self.recorded_masked_key_presses.add(key)
                                    # 一次性详细日志：第X ms监听到 XX键
                                    if key not in self.monitor_logged_keys:
                                        elapsed_ms = (time.perf_counter() - self.cooldown_monitor_start_ts) * 1000
                                        print(f"[监控] 第{elapsed_ms:.1f} ms监听到 {key} 键按下")
                                        self.monitor_logged_keys.add(key)
                            except Exception:
                                pass
                    time.sleep(0.002)
                except Exception:
                    time.sleep(0.005)

        self.cooldown_press_monitor_thread = threading.Thread(target=worker, daemon=True)
        self.cooldown_press_monitor_thread.start()

    def stop_cooldown_press_monitor(self):
        """停止冷却期间的按键监控线程。"""
        if self.cooldown_press_monitor_running:
            self.cooldown_press_monitor_running = False
            print("[监控] 冷却期按键监控停止")

    def begin_cooldown_monitor_window(self, trigger_key):
        """从按键按下开始到冷却结束的监听窗口（覆盖0.05s预屏蔽期）。"""
        if self.is_locked_in:
            print(f"[锁定] 忽略开启监听窗口: {trigger_key}")
            return
        # 重叠检测：在切换 cooldown_active 之前判定，避免连触发丢失上一窗口录键
        overlap = bool(self.masked_keys) or self.cooldown_active
        self.cooldown_active = True
        # 计算预屏蔽窗口截止时间（当前时间+0.05s）
        self.cooldown_pre_mask_deadline = time.perf_counter() + 0.05
        self.cooldown_monitor_start_ts = time.perf_counter()
        self.monitor_logged_keys.clear()
        print(f"[监控] 监听窗口开始（包含预屏蔽0.05s），触发键：{trigger_key}")
        # 设定预屏蔽监听集合
        if trigger_key == 'k':
            # k监听n
            targets = ['n']
        elif trigger_key == 'o':
            # o监听n,k
            targets = ['n', 'k']
        elif trigger_key in ('i', 'l'):
            # i/l监听k,n
            targets = ['k', 'n']
        elif trigger_key in ('n', 'm'):
            # n/m不监听任何
            targets = []
        else:
            targets = COOLDOWN_KEYS
        try:
            with self.recorded_keys_lock:
                # 仅在没有重叠的冷却时清空录键；否则保留前一窗口未消费的按键
                if not overlap:
                    self.recorded_masked_key_presses.clear()
            if overlap:
                # 叠加目标集合，避免覆盖上一触发的监听键
                self.cooldown_monitor_targets = self.cooldown_monitor_targets | set(targets)
            else:
                self.cooldown_monitor_targets = set(targets)
        except Exception:
            self.cooldown_monitor_targets = set(targets)
        # 启动监控线程（如果未启动）
        self.start_cooldown_press_monitor()

    def refresh_cooldown_monitor_targets(self, target_keys):
        """在物理屏蔽开始时刷新监听集合，同时维持记录。"""
        if self.is_locked_in:
            print("[锁定] 忽略刷新监听集合")
            return
        try:
            self.cooldown_monitor_targets = set(target_keys)
        except Exception:
            self.cooldown_monitor_targets = set(target_keys)
        print(f"[监控] 刷新监听集合：{sorted(list(self.cooldown_monitor_targets))}")

    def initialize_kmbox(self):
        """初始化kmbox连接"""
        try:
            kmNet.init('192.168.2.188', '3078', '1C763CAB')
            kmNet.monitor(5001)
            return True
        except Exception as e:
            print(f"kmbox连接失败: {e}")
            return False
    
    def initialize_model(self):
        """初始化YOLO模型"""
        print(f"加载模型: {CONFIG['model_weights']}")
        
        # 使用与确认V2.py相同的模型加载方式
        self.model, self.device, self.half, self.stride, self.names = get_model(CONFIG['model_weights'])
        self.imgsz = check_img_size(IMGSZ, s=self.stride)
        
        # 模型优化设置
        self.model.eval()
        if hasattr(self.model, 'fuse'):
            self.model.fuse()
        
        print(f"模型加载完成: 设备={self.device}")

        # 固定将 hit 类别设为索引2（与特瑞.py保持一致）
        self.hit_class_ids = {2}
        try:
            mapped = [self.names[i] for i in sorted(self.hit_class_ids)]
        except Exception:
            mapped = list(sorted(self.hit_class_ids))
        print(f"hit类别固定为索引{sorted(self.hit_class_ids)} 名称={mapped}")

        # 模型预热：dummy forward 触发 cuDNN benchmark / kernel JIT，避免首帧延迟数秒
        try:
            t0 = time.perf_counter()
            dummy = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)
            if self.half:
                dummy = dummy.half()
            with torch.no_grad():
                for _ in range(3):
                    self.model(dummy)
            if torch.cuda.is_available() and 'cpu' not in str(self.device):
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            print(f"[预热] 模型 warmup 完成，用时 {(time.perf_counter()-t0)*1000:.0f} ms")
        except Exception as e:
            print(f"[预热] 模型 warmup 失败: {e}")

    def capture_screenshot(self):
        """使用mss快速截图"""
        screenshot = self.mss_instance.grab(MONITOR)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    
    def preprocess_image(self, img0):
        """优化的图像预处理"""
        img = letterbox(img0, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device, non_blocking=True)
        img = img.half() if self.half else img.float()
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]
        return img
    
    def detect_objects(self, img_tensor):
        """优化的目标检测"""
        with torch.no_grad():
            pred = self.model(img_tensor, augment=False, visualize=False)[0]
            
            # NMS处理
            pred = non_max_suppression(
                pred,
                CONFIG['conf_threshold'],
                CONFIG['iou_threshold'],
                CONFIG['classes'],
                False,
                max_det=CONFIG['max_det']
            )
            
            return pred[0]  # 返回第一个图像的检测结果
    
    def analyze_positions(self, detections):
        """按特瑞.py逻辑：检测class2(hit)在上半部分(Y≤104)，使用固定X分界(958)，class1由用户设定侧。"""
        # 固定边界（特瑞.py）
        hit_horizontal_boundary = 104  # 只保留Y<=104的hit
        left_right_vertical_boundary = 958  # X分界

        class0_positions = []  # hit位置（这里用class2作为hit）
        class1_positions = []  # 用户设定位置（占位，保持签名一致）

        if detections is not None and len(detections) > 0:
            for *xyxy, conf, cls in detections:
                cls_id = int(cls)
                confidence = float(conf)
                x_center = (float(xyxy[0]) + float(xyxy[2])) / 2
                y_center = (float(xyxy[1]) + float(xyxy[3])) / 2

                # 使用名称映射的hit类别集合
                if cls_id in getattr(self, 'hit_class_ids', {2}):
                    if y_center <= hit_horizontal_boundary:
                        class0_positions.append({
                            'x_center': x_center,
                            'confidence': confidence,
                            'side': 'left' if x_center < left_right_vertical_boundary else 'right'
                        })

            # 仅在首次检测到hit时输出一次坐标信息
            if class0_positions and not hasattr(self, '_hit_coords_logged'):
                print(f"\n=== 检测到hit目标 ===")
                for i, pos in enumerate(class0_positions):
                    print(f"  hit{i+1}: x={pos['x_center']:.0f}, y≤{hit_horizontal_boundary}, 位置={pos['side']}")
                self._hit_coords_logged = True

        return class0_positions, class1_positions, left_right_vertical_boundary

    def should_trigger_output(self, class0_positions, class1_positions):
        """判断是否应该触发按键输出"""
        # 如果没有检测到class0，不输出
        if not class0_positions:
            return False, "无class0目标"
        
        # 如果用户没有设定class1位置，不输出
        if self.user_set_class1_side is None:
            return False, "用户未设定class1位置（请按Z键设为左边或C键设为右边）"
        
        # 获取最佳class0位置
        best_class0 = max(class0_positions, key=lambda x: x['confidence'])
        class0_side = best_class0['side']
        
        # 使用用户设定的class1位置（不再依赖yolo检测）
        class1_side = self.user_set_class1_side
        
        user_setting = f"用户设定class1位置:{class1_side}"
        
        # 只有在对立面时才输出
        if class0_side != class1_side:
            return True, f"对立面：class0({class0_side}) vs class1({class1_side}), {user_setting}"
        else:
            return False, f"同边：class0({class0_side}) class1({class1_side}), {user_setting}"

    def process_frame_for_key(self, key_name, detections):
        """为特定按键处理检测结果 - 新的位置判断逻辑"""
        key_config = KEY_CONFIGS[key_name]
        key_state = self.key_states[key_name]
        
        # 分析位置关系
        class0_positions, class1_positions, center_x = self.analyze_positions(detections)
        
        # 判断是否应该触发输出
        should_trigger, reason = self.should_trigger_output(class0_positions, class1_positions)
        
        if should_trigger and not key_state['key_triggered']:
            # 记录检测成功时间
            self.log_timing(key_name, "检测成功")
            
            # 执行按键输出
            best_class0 = max(class0_positions, key=lambda x: x['confidence'])
            self.send_key_output(key_name, best_class0['x_center'], best_class0['confidence'], reason)
            key_state['no_detection_frames'] = 0

            # 进入锁定模式
            print(f"🔒 进入锁定模式，等待Class 0消失...")
            self.is_locked_in = True
            # 记录进入锁定模式时间
            self.log_timing(key_name, "进入锁定模式")
            # 重置所有按键的检测状态
            for k in self.key_states:
                self.key_states[k]['detection_active'] = False
                self.key_states[k]['key_triggered'] = False
                self.key_states[k]['no_detection_frames'] = 0

        elif class0_positions:  # 有class0但不触发
            key_state['no_detection_frames'] = 0
            if self.frame_count % 30 == 0:  # 每30帧显示一次原因
                print(f"[{key_name.upper()}] {reason}")
        else:
            # 没有检测到任何目标
            key_state['no_detection_frames'] += 1
            
            # 如果连续多帧没有检测到目标，停止检测
            if key_state['no_detection_frames'] >= key_config['max_no_detection_frames']:
                print(f"*** {key_name.upper()}键检测：连续{key_config['max_no_detection_frames']}帧无目标，停止检测 ***")
                key_state['detection_active'] = False
                key_state['key_triggered'] = False
                key_state['no_detection_frames'] = 0

    def send_key_output(self, key_name, x_center, confidence, reason):
        """发送按键输出（N/I/O/L：先解除屏蔽→输出记录到的按键；其他键不输出）"""
        try:
            print(f"\n=== {key_name.upper()}键检测成功 ===")
            print(f"hit坐标: x={x_center:.0f}, 置信度={confidence:.2f}")
            print(f"触发原因: {reason}")
            
            if key_name in ('n', 'i', 'o', 'l'):
                # 最高优先级：先解除所有屏蔽
                #time.sleep(random.uniform(0.09,0.097))
                self.cancel_all_cooldowns_and_unmask()
                # 读取并清空记录到的被屏蔽期间按下的键（包含space）
                try:
                    with self.recorded_keys_lock:
                        keys_to_emit = list(self.recorded_masked_key_presses)
                        self.recorded_masked_key_presses.clear()
                except Exception:
                    keys_to_emit = []
                if keys_to_emit:
                    emitted = []
                    # 同时按下所有按键
                    for key in keys_to_emit:
                        try:
                            hid = CONFIG['key_codes'][key]
                        except KeyError:
                            continue
                        try:
                            kmNet.keydown(hid)
                            emitted.append(key)
                        except Exception as e:
                            print(f"✗ 按下按键{key}失败: {e}")
                    
                    # 保持按键按下状态一段时间
                    if emitted:
                        # 这里是为了模拟人类按键按下的持续时间，随机延迟0.01到0.05秒
                        time.sleep(random.uniform(0.03, 0.08))
                        
                        # 同时松开所有按键
                        for key in emitted:
                            try:
                                hid = CONFIG['key_codes'][key]
                                kmNet.keyup(hid)
                            except Exception as e:
                                print(f"✗ 松开按键{key}失败: {e}")
                        
                        print(f"✓ {key_name.upper()}键同时输出记录按键: {','.join(emitted)}")
                        self.log_timing(key_name, "按键输出完成")
                        # 设置按键冷却（仅在有输出时）
                        current_time = time.perf_counter()
                        self.key_output_cooldown[key_name] = current_time + self.cooldown_duration
                        print(f"  {key_name.upper()}键冷却{self.cooldown_duration}秒")
                    else:
                        print("(无可输出的记录按键)")
                else:
                    print("(冷却期间未检测到按键按下，不输出)")
            else:
                # 其他按键不输出
                print(f"  {key_name.upper()}键无输出")
                # 立即终止所有冷却（解除物理屏蔽）
                self.cancel_all_cooldowns_and_unmask()
                print("✓ 已终止所有按键冷却期")
                # 记录冷却终止时间
                self.log_timing(key_name, "冷却终止完成")

            # 统计信息
            self.key_states[key_name]['detection_count'] += 1
            self.total_detection_count += 1
            self.key_states[key_name]['key_triggered'] = True
        except Exception as e:
            print(f"按键操作失败: {e}")
            import traceback
            traceback.print_exc()  # 添加详细的错误信息输出

    def process_frame(self):
        """处理单帧 - 为所有活跃的按键检测"""
        start_time = time.perf_counter()
        
        try:
            # 更新kmNet物理屏蔽冷却状态
            self.update_cooldowns()
            active_keys = [k for k, state in self.key_states.items() if state['detection_active']]
            
            # 如果没有任何活动键，并且不处于锁定模式，则可以提前返回
            if not active_keys and not self.is_locked_in:
                return

            img_orig = None
            detections = None

            def capture_and_detect_once():
                """辅助函数，确保每帧只截图和检测一次"""
                nonlocal img_orig, detections
                if img_orig is None:
                    img_orig = self.capture_screenshot()
                    img_tensor = self.preprocess_image(img_orig)
                    detections = self.detect_objects(img_tensor)
                    if detections is not None and len(detections):
                        detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], img_orig.shape).round()

            # 捕获一次用于后续各键判断
            capture_and_detect_once()

            # 1. 优先处理M键（如果激活）
            if 'm' in active_keys and not self.is_locked_in:
                self.process_frame_for_key('m', detections)
                active_keys.remove('m')

            # 2. 处理锁定模式或其他按键
            if self.is_locked_in:
                # 在锁定模式下，只检查class0是否消失
                class0_positions, _, _ = self.analyze_positions(detections)
                if not class0_positions:
                    print("🔓 Class 0消失，解除锁定，返回监听模式")
                    self.is_locked_in = False
                    # 找到触发锁定的按键并停止计时器
                    for k, state in self.key_states.items():
                        if state.get('key_triggered', False):
                            self.stop_key_timer(k)
                            break
            
            elif active_keys: # 如果不在锁定模式，且还有其他活动按键
                for key_name in active_keys:
                    self.process_frame_for_key(key_name, detections)

            # 每20帧显示一次状态
            if self.frame_count % 20 == 0:
                processing_time = (time.perf_counter() - start_time) * 1000
                fps = 1000 / processing_time if processing_time > 0 else 0
                # 重新获取最新的活动按键数量用于显示
                current_active_keys = [k for k, state in self.key_states.items() if state['detection_active']]
                active_count = len(current_active_keys)
                lock_status = "LOCKED" if self.is_locked_in else "LISTENING"
                print(f"[{self.frame_count}] FPS:{fps:.0f} Status:{lock_status} ActiveKeys:{active_count} TotalTriggers:{self.total_detection_count}")
            
            self.frame_count += 1
            
        except Exception as e:
            import traceback
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!! FATAL ERROR in process_frame: {e} !!!!!!!!!")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def trigger_key_function(self, key_name):
        """辅助函数：激活指定按键的检测功能"""
        key_state = self.key_states.get(key_name)
        if key_state and not key_state['detection_active']:
            key_state['detection_active'] = True
            key_state['key_triggered'] = False
            key_state['no_detection_frames'] = 0
            
            # 修改：如果是m键、k键、o键、l键、i键或n键，立即检测W键状态（用于自动触发的情况）
            if key_name in ['m', 'k', 'o', 'l', 'i', 'n']:  # 删除模式检查，因为只有3键模式
                print(f"\n[DEBUG] 自动触发{key_name}键，检测W键状态...")
                try:
                    import win32api
                    VK_W = 0x57
                    # 只检测一次w键状态并存储
                    w_state = win32api.GetAsyncKeyState(VK_W) & 0x8000 != 0
                    key_state['w_key_state'] = w_state  # 存储到按键状态中
                    print(f"\n[{key_name.upper()}键自动触发] 记录W键状态: {'按下' if w_state else '松开'} - {time.perf_counter():.3f}")
                    print(f"[DEBUG] 自动触发win32api检测结果: {w_state}")
                    print(f"[DEBUG] 自动触发全局状态: {self.w_key_state}")
                except Exception as e:
                    print(f"自动触发获取W键状态失败: {e}")
                    w_state = keyboard.is_pressed('w')
                    key_state['w_key_state'] = w_state
                    print(f"[DEBUG] 自动触发keyboard库检测结果: {w_state}")
            

            print(f"--- 已自动触发 {key_name.upper()} 键功能 ---")

    def listen_for_key(self, key_name):
        """辅助函数：监听单个按键的触发"""
        key_config = KEY_CONFIGS[key_name]
        key_state = self.key_states[key_name]
        trigger_key = key_config['trigger_key']
        
        # 检查是否在冷却期间 - 修复：检查触发键本身的冷却时间
        current_time = time.perf_counter()
        cooldown_end_time = self.key_output_cooldown.get(trigger_key, 0)
        
        if current_time < cooldown_end_time:
            # 在冷却期间，跳过检测
            if self.frame_count % 60 == 0:  # 每60帧显示一次冷却状态
                remaining_time = cooldown_end_time - current_time
                print(f"[{trigger_key.upper()}键冷却中，剩余{remaining_time:.1f}秒]")
            key_state['last_trigger_state'] = keyboard.is_pressed(trigger_key)
            return
        
        current_trigger_state = keyboard.is_pressed(trigger_key)
        if current_trigger_state and not key_state['last_trigger_state']:
            if not key_state['detection_active']:
                # 修复：在按键触发的瞬间立即记录W键状态
                if key_name in ['m', 'k', 'o', 'l', 'i', 'n']:  # 删除模式检查，因为只有3键模式
                    try:
                        import win32api
                        VK_W = 0x57
                        # 立即检测并记录w键状态
                        w_state = win32api.GetAsyncKeyState(VK_W) & 0x8000 != 0
                        key_state['w_key_state'] = w_state
                        print(f"\n[{key_name.upper()}键触发瞬间] 立即记录W键状态: {'按下' if w_state else '松开'} - {time.perf_counter():.3f}")
                        print(f"[DEBUG] 触发瞬间win32api检测结果: {w_state}")
                        print(f"[DEBUG] 触发瞬间全局状态: {self.w_key_state}")
                    except Exception as e:
                        print(f"触发瞬间获取W键状态失败: {e}")
                        w_state = keyboard.is_pressed('w')
                        key_state['w_key_state'] = w_state
                        print(f"[DEBUG] 触发瞬间keyboard库检测结果: {w_state}")
                
                key_state['detection_active'] = True
                key_state['key_triggered'] = False
                key_state['no_detection_frames'] = 0

                print(f"\n*** {trigger_key.upper()}键触发！开始位置关系检测 ***")
                # 仅对真实用户触发应用kmNet物理冷却（延迟0.01s）
                if key_name in TRIGGER_COOLDOWN_S:
                    try:
                        import threading
                        def delayed_apply():
                            try:
                                time.sleep(0.01)
                                self.apply_cooldown_for_trigger(key_name)
                            except Exception as e:
                                print(f"延迟应用冷却失败: {e}")
                        threading.Thread(target=delayed_apply, daemon=True).start()
                    except Exception as e:
                        print(f"启动延迟冷却线程失败: {e}")
                time.sleep(0.09)
        key_state['last_trigger_state'] = current_trigger_state

    def start_w_key_monitor(self):
        """启动w键状态监控线程"""
        import threading
        import win32api
        
        # 定义W键的虚拟键码
        VK_W = 0x57  # W键的虚拟键码
        
        def monitor_w_key():
            last_state = False
            while self.w_key_monitor_running:
                try:
                    # 使用正确的虚拟键码
                    current_state = win32api.GetAsyncKeyState(VK_W) & 0x8000 != 0
                    if current_state != last_state:
                        self.w_key_state = current_state
                        print(f"\n[W键状态] {'按下' if current_state else '松开'} - {time.perf_counter():.3f}")
                        last_state = current_state
                except Exception as e:
                    print(f"W键监控错误: {e}")
                time.sleep(0.001)
        
        if not self.w_key_monitor_running:
            self.w_key_monitor_running = True
            self.w_key_thread = threading.Thread(target=monitor_w_key, daemon=True)
            self.w_key_thread.start()
            print("W键状态监控已启动（实时优化版）")

    def stop_w_key_monitor(self):
        """停止w键状态监控线程"""
        self.w_key_monitor_running = False
        # 线程为daemon，将很快退出

    def prewarm_resources(self):
        """启动期预热：连接 kmbox、创建 mss、加载并 warmup 模型、跑通截图管线。
        非致命失败不会阻断启动，start_system 仍会按需兜底重试。"""
        print("\n[预热] 加载模型与初始化设备（首次启动可能耗时数秒）...")
        if not self.kmbox_ready:
            if self.initialize_kmbox():
                self.kmbox_ready = True
                print("[预热] kmbox 已连接")
            else:
                print("[预热] kmbox 连接失败，按 1 时会再次尝试")
        if self.mss_instance is None:
            try:
                self.mss_instance = mss.mss()
                print("[预热] 截图实例已就绪")
            except Exception as e:
                print(f"[预热] mss 初始化失败: {e}")
        if self.model is None:
            try:
                self.initialize_model()
            except Exception as e:
                print(f"[预热] 模型加载失败: {e}")
        if self.model is not None and self.mss_instance is not None:
            try:
                t0 = time.perf_counter()
                img_orig = self.capture_screenshot()
                img_tensor = self.preprocess_image(img_orig)
                with torch.no_grad():
                    _ = self.detect_objects(img_tensor)
                print(f"[预热] 截图+推理管线用时 {(time.perf_counter()-t0)*1000:.0f} ms")
            except Exception as e:
                print(f"[预热] 截图管线预热失败: {e}")
        print("[预热] 完成。按 1 开启系统。")

    def start_system(self):
        if self.system_active:
            return
        print("\n>>> 启动系统：初始化设备与模型，并注册事件...")
        # 初始化kmbox
        if not self.kmbox_ready:
            print("初始化kmbox连接...")
            if not self.initialize_kmbox():
                print("kmbox初始化失败，启动中止")
                return
            self.kmbox_ready = True
            print("kmbox初始化成功")
        # 初始化mss
        if self.mss_instance is None:
            self.mss_instance = mss.mss()
        # 初始化模型
        if self.model is None:
            self.initialize_model()
        # 启动W键监控
        self.start_w_key_monitor()
        # 注册按键事件
        self.register_key_event_handlers()
        # 状态
        self.system_active = True
        self.is_locked_in = False
        print(">>> 系统已启动")

    def stop_system(self):
        if not self.system_active:
            return
        print("\n>>> 停止系统：注销事件并停止监控...")
        # 关闭检测与锁定
        self.system_active = False
        self.is_locked_in = False
        for key_name in self.key_states:
            self.key_states[key_name]['detection_active'] = False
            self.key_states[key_name]['key_triggered'] = False
            self.key_states[key_name]['no_detection_frames'] = 0
        # 释放所有冷却屏蔽，避免停系统时物理按键被卡
        self.cancel_all_cooldowns_and_unmask()
        # 清空未消费的录键，避免下次启动遗留旧数据
        try:
            with self.recorded_keys_lock:
                self.recorded_masked_key_presses.clear()
            self.cooldown_monitor_targets = set()
        except Exception:
            pass
        # 注销事件
        self.unregister_key_event_handlers()
        # 停止W键监控
        self.stop_w_key_monitor()
        print(">>> 系统已停止")

    def run(self):
        """运行检测器"""
        print("=== 多按键智能位置检测器 ===")
        print("1键开启系统 | R键关闭系统")
        print("Z键设定Class1在左边 | C键设定Class1在右边")
        print("T键切换计时显示 | 计时功能已启用")
        print("\n检测逻辑：")
        print("- 同时检测class0(hit)和class1(me)的位置")
        print("- 忽略截图下半部分的class0")
        print("- Class1位置由用户通过Z键(左)或C键(右)手动设定")
        print("- 以截图中心线为界：左边 vs 右边")
        print('- class0和class1在对立面时 → 输出按键，并进入"锁定模式"')
        print('- 锁定模式下，等待class0从画面消失后，才返回"监听模式"')
        print("\n全局智能宏（无需开启系统；V键切换启用/禁用，禁用后可正常打字）:")
        print("  J键: W保持20~40ms后点击K")
        print("  M键: 附加W保持30~60ms")
        print("  A键: 智能宏（原宏→临时宏→临时宏输出）")
        print("  D键: 智能宏（原宏→临时宏→临时宏输出）")
        print("  Space键: 智能宏（原宏→临时宏→临时宏输出，仅按W键，不输出其他按键）")
        print("  逗号键: 复杂按键序列（松开逗号键可停止）")
        print("\n系统开启后，支持以下按键:")
        for key_name, config in KEY_CONFIGS.items():
            print(f"  {config['description']}")
        print("按Ctrl+0退出")

        # 启动期预热：避免按 1 后才同步加载模型，阻塞键盘轮询导致首次启动看似"无响应"
        self.prewarm_resources()

        exit_key_start_time = 0
        last_on_state = False   # 2键
        last_off_state = False  # R键
        last_on_time = 0
        last_off_time = 0
        control_key_debounce = 0.2  # 控制键防抖动200ms
        
        # 新增：z键、c键和t键的状态跟踪
        last_z_state = False   # Z键
        last_c_state = False   # C键
        last_t_state = False   # T键
        last_v_state = False   # V键（切换全局宏，拼音中不使用）
        last_z_time = 0
        last_c_time = 0
        last_t_time = 0
        last_v_time = 0
        
        try:
            while True:
                current_time = time.perf_counter()
                
                # 检查退出
                if keyboard.is_pressed('ctrl+0'):
                    if exit_key_start_time == 0:
                        exit_key_start_time = current_time
                        print("\r按住Ctrl+0退出...", end="")
                    elif current_time - exit_key_start_time >= 1:
                        print("\n正在退出...")
                        break
                else:
                    exit_key_start_time = 0
                
                # 检查1键开启系统（作为全局启动开关）
                current_on_state = keyboard.is_pressed('1')
                if current_on_state and not last_on_state and current_time - last_on_time > control_key_debounce:
                    self.start_system()
                    last_on_time = current_time
                    time.sleep(0.1)
                last_on_state = current_on_state
                
                # 检查R键关闭系统（作为全局停止开关，不退出脚本）
                current_off_state = keyboard.is_pressed('r')
                if current_off_state and not last_off_state and current_time - last_off_time > control_key_debounce:
                    self.stop_system()
                    last_off_time = current_time
                    time.sleep(0.1)
                last_off_state = current_off_state
                
                # 新增：检查Z键设定class1在左边
                current_z_state = keyboard.is_pressed('z')
                if current_z_state and not last_z_state and current_time - last_z_time > control_key_debounce:
                    self.user_set_class1_side = 'left'
                    print(f"\n*** Z键按下！Class1位置设定为左边 ***")
                    last_z_time = current_time
                    time.sleep(0.1)
                last_z_state = current_z_state
                
                # 新增：检查C键设定class1在右边
                current_c_state = keyboard.is_pressed('c')
                if current_c_state and not last_c_state and current_time - last_c_time > control_key_debounce:
                    self.user_set_class1_side = 'right'
                    print(f"\n*** C键按下！Class1位置设定为右边 ***")
                    last_c_time = current_time
                    time.sleep(0.1)
                last_c_state = current_c_state
                
                # 新增：检查T键切换计时显示
                current_t_state = keyboard.is_pressed('t')
                if current_t_state and not last_t_state and current_time - last_t_time > control_key_debounce:
                    self.timing_enabled = not self.timing_enabled
                    status = "启用" if self.timing_enabled else "禁用"
                    print(f"\n*** T键按下！计时功能已{status} ***")
                    last_t_time = current_time
                    time.sleep(0.1)
                last_t_state = current_t_state

                # 新增：检查V键切换全局宏（拼音中不使用 v，关闭后 j/m/a/d/逗号 可正常打字）
                current_v_state = keyboard.is_pressed('v')
                if current_v_state and not last_v_state and current_time - last_v_time > control_key_debounce:
                    self.global_macros_enabled = not self.global_macros_enabled
                    status = "启用" if self.global_macros_enabled else "禁用（可正常打字）"
                    print(f"\n*** V键按下！全局宏已{status} ***")
                    last_v_time = current_time
                    time.sleep(0.1)
                last_v_state = current_v_state

                # 主循环中的按键监听（事件模式下不再轮询k/l/m/n/o/i）
                # 事件模式：不进行任何按键轮询
                
                # 主检测循环
                if self.system_active:
                    self.process_frame()
                else:
                    # 当系统关闭时，减小CPU占用
                    time.sleep(0.01)
                
                # 添加主循环延迟，避免过度频繁检测
                time.sleep(0.002)  # 2ms延迟，平衡性能和响应
                
        except KeyboardInterrupt:
            print("\n程序被中断")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        try:
            # 停止w键监控线程
            if self.w_key_thread and self.w_key_thread.is_alive():
                self.w_key_thread = None  # 线程是daemon线程，主程序退出时会自动结束
            kmNet.unmask_all()
        except:
            pass

if __name__ == "__main__":
    print("=== 智能位置检测系统 ===")
    print("新逻辑：根据class0和class1的位置关系判断是否输出按键")
    print("截图区域: 1070x60像素，中心线位置: 535像素")
    print("支持M,K,L,U,I,O键的统一检测逻辑")
    
    # 创建检测器
    detector = MultiKeyDetector()
    
    # 运行检测
    detector.run() 