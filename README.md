# Sagat.py 移植说明

本仓库只包含运行 `Sagat.py` 所需的文件。

## 文件说明

| 文件 / 目录 | 作用 |
|---|---|
| `Sagat.py` | 主脚本 |
| `MyTrainFor_SF6.pt` | YOLO 模型权重（脚本按相对路径加载，必须与脚本同目录） |
| `models/` | YOLO 推理代码（config、load_model、general 等） |
| `kmNet.cp310-win_amd64.pyd` | kmbox 驱动模块，对应 Python 3.10 |
| `kmbox/python_pyd/` | 其他 Python 版本的 kmNet 驱动（cp37 ~ cp313） |
| `requirements.txt` | Python 依赖 |
| `启动.bat` | 启动菜单，使用 `.conda\Scripts\python.exe` |

## 环境要求

- Windows + NVIDIA 显卡（代码强制要求 CUDA，无 GPU 会直接报错）
- Python 3.10（如用其他版本，把 `kmbox/python_pyd/` 里对应版本的 `kmNet.*.pyd` 复制到根目录）
- 需要以管理员身份运行（`keyboard` 库和 kmNet 物理按键屏蔽需要）

## 安装步骤

```bat
:: 在仓库根目录创建虚拟环境（启动.bat 默认找 .conda）
python -m venv .conda
.conda\Scripts\activate

:: 安装 PyTorch（CUDA 12.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

:: 安装其余依赖
pip install -r requirements.txt
```

原机器上的版本供参考：Python 3.10.0，torch 2.11.0+cu128，opencv-python 4.12，numpy 2.2.6。

## 运行

双击 `启动.bat` 选择 `Sagat.py`，或：

```bat
.conda\Scripts\python.exe Sagat.py
```

## 注意

`Sagat.py` 中 `initialize_kmbox()` 写死了 kmbox 的 IP、端口和 UUID
（`192.168.2.188` / `3078` / `1C763CAB`）。如果新电脑连接的 kmbox 设备或网段不同，需要修改这一行。
