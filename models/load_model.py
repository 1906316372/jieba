import torch
from models.config import *
from models.experimental import attempt_load
from models.torch_utils import select_device


def get_model(weights=None):
    # 选择设备；GPU还是CPU
    device = select_device('')  # device 设备
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # 使用传入的weights参数，如果没有则使用默认的WEIGHTS
    model_weights = weights if weights is not None else WEIGHTS
    
    # 必须在 GPU 上加载模型，不允许回退到 CPU
    if device.type != 'cuda':
        raise RuntimeError(
            "错误: 设备不是CUDA。程序要求必须使用GPU，不允许CPU模式。\n"
            "请检查GPU是否可用，或安装支持您GPU的PyTorch版本。"
        )
    
    try:
        model = attempt_load(model_weights, map_location=device)
        # 测试 GPU 是否真的可用（尝试创建一个简单的 tensor）
        test_tensor = torch.zeros(1).to(device)
        del test_tensor
    except (RuntimeError, AssertionError) as e:
        error_msg = str(e)
        # 检查是否是 CUDA capability 不兼容错误
        if 'no kernel image' in error_msg or 'CUDA capability' in error_msg or 'sm_' in error_msg:
            raise RuntimeError(
                f"GPU 不兼容当前 PyTorch 版本 {torch.__version__}。\n"
                f"RTX 5070 Ti (sm_120) 需要 PyTorch nightly 版本。\n"
                f"请运行以下命令安装支持 sm_120 的 PyTorch:\n"
                f"pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128\n"
                f"错误详情: {error_msg}"
            ) from e
        else:
            # 其他错误，直接抛出
            raise

    stride = int(model.stride.max())  # model stride
    
    # 确保模型使用正确的类别名称
    if hasattr(model, 'module'):
        model.module.names = ['hit', 'me']
    else:
        model.names = ['hit', 'me']
    
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        try:
            model.half()  # to FP16
        except RuntimeError as e:
            # 如果 half precision 失败，使用 float32
            print(f"警告: 无法使用 FP16，使用 FP32: {e}")
            half = False

    return model, device, half, stride, names


