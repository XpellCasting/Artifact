"""Selección explícita de CPU o GPU con la interfaz común CUDA/HIP de PyTorch."""
import torch


def training_device(requested):
    device = torch.device(requested)
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError("Dispositivo de entrenamiento inválido: usa cpu o cuda:N")
    if not torch.cuda.is_available():
        raise ValueError(
            "GPU no disponible en PyTorch. Para AMD en WSL2 instala PyTorch con ROCm "
            "y comprueba torch.version.hip y torch.cuda.is_available(); "
            "para ejecutar en CPU selecciona device='cpu'.")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"GPU cuda:{index} no disponible; detectadas {torch.cuda.device_count()}")
    return torch.device("cuda", index)


def device_metadata(device):
    return {
        "device": str(device),
        "backend": "rocm" if device.type == "cuda" and torch.version.hip else device.type,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }
