"""Comprueba dependencias y ejecuta un paso real GAT-LSTM en CPU o CUDA."""
from __future__ import annotations

import argparse
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def _expected_versions():
    """Read project pins without importing optional neural dependencies."""
    versions = {}
    for line in (ROOT / "backend" / "requirements-gat.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "==" in line:
            name, version = line.strip().split("==", 1)
            versions[name] = version
    return versions


def _smoke_test(torch, device):
    from app.modulos.clasificador.config import Config
    from app.modulos.clasificador.network_segments import SegmentGATLSTM
    from app.modulos.clasificador.segments import BASE_NAMES, FEATURE_NAMES, MAX_POINTS, MAX_SEGMENTS

    torch.manual_seed(42)
    config = Config(dropout=0, lstm_hidden=8, device=str(device))
    kinds = ["continuous", "multistate"]
    model = SegmentGATLSTM(len(FEATURE_NAMES), kinds, config).to(device)
    x = torch.zeros(1, 3 * MAX_SEGMENTS, 2, len(FEATURE_NAMES), device=device)
    x[..., 0] = torch.linspace(0, 1, 3 * MAX_SEGMENTS, device=device).view(1, -1, 1)
    category_start = len(BASE_NAMES) + 2 * MAX_POINTS
    x[..., category_start:category_start + MAX_POINTS] = 1
    for scale in range(3):
        for segment in range(MAX_SEGMENTS):
            x[:, scale * MAX_SEGMENTS + segment, :, -MAX_SEGMENTS + segment] = 1
    states = torch.zeros(x.shape[:3], dtype=torch.long, device=device)
    present = torch.ones_like(states, dtype=torch.bool)
    adjacency = torch.eye(2, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    logits, fault_logits = model(x, states, adjacency, present)
    target = torch.tensor([0, 2], device=device)
    fault_target = torch.tensor([0, 1], device=device)
    loss = (torch.nn.functional.cross_entropy(logits.reshape(-1, 3), target)
            + torch.nn.functional.cross_entropy(fault_logits.reshape(2, -1), fault_target))
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("La pérdida del paso de prueba no es finita")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("El backward produjo gradientes ausentes o no finitos")
    for head in (model.classifier, model.fault_type_classifier):
        if head.weight.grad is None or not bool(head.weight.grad.abs().sum() > 0):
            raise RuntimeError("Una cabeza de clasificación no recibió gradientes")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = tuple(output.cpu() for output in model(x, states, adjacency, present))
    # Exercise the same portability used by candidates: CPU state_dict, safe load.
    buffer = io.BytesIO()
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, buffer)
    buffer.seek(0)
    restored = SegmentGATLSTM(len(FEATURE_NAMES), kinds, config).cpu()
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        actual = restored(x.cpu(), states.cpu(), adjacency.cpu(), present.cpu())
    if not all(torch.allclose(left, right, atol=1e-4, rtol=1e-3)
               for left, right in zip(expected, actual)):
        raise RuntimeError("La recarga en CPU no coincide con la inferencia del dispositivo elegido")
    return {
        "forward": True, "backward": True, "optimizer_step": True,
        "fault_type_head": True, "cpu_reload": True,
        "loss": float(loss.detach().cpu()),
        "classification_shape": list(logits.shape), "fault_type_shape": list(fault_logits.shape),
    }


def check_environment(requested="auto", require_cuda=False):
    report = {
        "ok": False, "requested_device": requested, "require_cuda": require_cuda,
        "python": platform.python_version(), "platform": platform.platform(),
        "expected_versions": _expected_versions(), "versions": {}, "warnings": [],
    }
    try:
        import torch
        report["versions"]["torch"] = torch.__version__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count()
        import torch_geometric
        report["versions"]["torch-geometric"] = torch_geometric.__version__
        for name, expected in report["expected_versions"].items():
            installed = report["versions"].get(name)
            if installed is None:
                try:
                    installed = metadata.version(name)
                except metadata.PackageNotFoundError:
                    report["warnings"].append(f"Dependencia ausente: {name}=={expected}")
                    continue
            if installed.split("+", 1)[0] != expected:
                report["warnings"].append(f"{name}: instalado {installed}; versión del proyecto {expected}")
        from app.modulos.clasificador.training import resolve_device

        device = torch.device(resolve_device(requested))
        report["selected_device"] = str(device)
        if require_cuda and device.type != "cuda":
            raise ValueError("Se requiere CUDA, pero el dispositivo elegido es CPU. Usa --device cuda.")
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            report["gpu"] = {
                "index": index, "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": total, "free_memory_bytes": free,
            }
            report["cudnn_version"] = torch.backends.cudnn.version()
        elif requested == "auto":
            report["warnings"].append("CUDA no está disponible; auto seleccionó CPU. Usa --require-cuda en el equipo NVIDIA.")
        report["smoke_test"] = _smoke_test(torch, device)
        report["ok"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["help"] = (
            "Consulta training/README.md. Comprueba el entorno Python activo y python -m pip check; "
            "para NVIDIA, nvidia-smi y una rueda PyTorch CUDA compatible con el controlador. "
            "No uses requirements-gat-docker.txt (CPU) en el entorno CUDA."
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda o cuda:N (índice visible de GPU)")
    parser.add_argument("--require-cuda", action="store_true", help="Falla si el dispositivo seleccionado no es CUDA")
    parser.add_argument("--output", type=Path, help="Guarda además el informe JSON en esta ruta")
    args = parser.parse_args(argv)
    result = check_environment(args.device, args.require_cuda)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
