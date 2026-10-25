"""Prueba GAT, LSTM y retropropagación en CPU o GPU antes de entrenar."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def check_device(requested, require_rocm=False):
    import torch
    from torch.nn import functional as F
    from app.modulos.clasificador.config import Config
    from app.modulos.clasificador.devices import device_metadata, training_device
    from app.modulos.clasificador.network_segments import SegmentGATLSTM
    from app.modulos.clasificador.segments import FEATURE_NAMES, MAX_POINTS, MAX_SEGMENTS

    config = Config(device=requested, dropout=0)
    device = training_device(config.device)
    if require_rocm and (device.type != "cuda" or not torch.version.hip):
        raise ValueError("La prueba requiere una GPU AMD con PyTorch ROCm (torch.version.hip)")
    torch.manual_seed(config.seed)
    model = SegmentGATLSTM(len(FEATURE_NAMES), ["continuous", "binary", "multistate"], config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x = torch.randn(2, 3 * MAX_SEGMENTS, 3, len(FEATURE_NAMES), device=device)
    x[..., model.base_dim:model.base_dim + MAX_POINTS] = 1
    x[..., -MAX_SEGMENTS:] = 1  # Nonempty graphs exercise message passing.
    states = torch.zeros(2, 3 * MAX_SEGMENTS, 3, dtype=torch.long, device=device)
    present = torch.ones_like(states, dtype=torch.bool)
    adjacency = torch.eye(3, dtype=torch.bool, device=device)
    logits, fault_logits = model(x, states, adjacency, present)
    targets = torch.tensor([[0, 1, 2], [2, 0, 1]], device=device)
    loss = F.cross_entropy(logits.reshape(-1, 3), targets.flatten())
    loss = loss + F.cross_entropy(fault_logits.reshape(-1, fault_logits.size(-1)), targets.flatten())
    if not torch.isfinite(loss):
        raise FloatingPointError("La prueba produjo una pérdida no finita")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("La prueba produjo gradientes no finitos")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {**device_metadata(device), "status": "ok", "forward_backward": True, "loss": loss.item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-rocm", action="store_true", help="Exige GPU AMD y PyTorch con HIP")
    args = parser.parse_args()
    print(json.dumps(check_device(args.device, args.require_rocm), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
