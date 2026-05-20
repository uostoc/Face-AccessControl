# Face Access Control

面向书院场景的人脸识别门禁与晚归管理原型系统。

当前版本实现文档中的初期能力：

- SQLite 数据库初始化。
- 人员信息与人脸特征注册。
- 余弦相似度人员比对。
- 多帧投票确认。
- 陌生人员与晚归疑似事件判定。
- 摄像头实时识别主流程骨架。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.main init-db
```

## 常用命令

初始化数据库：

```powershell
python -m src.main init-db
```

注册人员基础信息：

```powershell
python -m src.main add-person --person-id STU2024001 --name 张三 --role student --department 光电信息学院
```

从图片生成并注册人脸特征：

```powershell
python -m src.main enroll-face --person-id STU2024001 --image data/registered_faces/zhangsan_front.jpg --angle front
```

启动摄像头识别：

```powershell
python -m src.main run-camera
```

显示摄像头预览窗口：

```powershell
python -m src.main run-camera --show-window
```

预览窗口中按 `q` 退出。

如果 CPU 识别卡顿，可以降低 `config/config.yaml` 中的 `recognition.detection_fps`，例如改成 `3` 到 `5`。

检查 ONNX Runtime 是否可用 GPU：

```powershell
python -m src.main check-gpu
```

如果输出包含 `CUDAExecutionProvider`，说明当前 venv 可以使用 NVIDIA GPU 推理。

## GPU 推理

GPU 推理需要 NVIDIA 显卡、可用显卡驱动、CUDA/cuDNN 运行库，以及 `onnxruntime-gpu`。

建议在 venv 中只保留一个 ONNX Runtime 包：

```powershell
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -r requirements-gpu.txt
python -m src.main check-gpu
```

`config/config.yaml` 默认优先使用 CUDA，失败时回退 CPU：

```yaml
recognition:
  providers:
    - CUDAExecutionProvider
    - CPUExecutionProvider
```

## 说明

`insightface` 和 `onnxruntime` 是实际人脸识别所需依赖。如果本机未安装，仍可以运行数据库、匹配、投票和事件判定测试。
