"""Compare PyG and dense training steps on prepared Desarrollo batches.

Does not save weights, activate models, or access Validation/Test partitions.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def benchmark(args):
    import torch
    from torch.nn import functional as F
    from app.modulos.clasificador.config import Config
    from app.modulos.clasificador.devices import device_metadata, training_device
    from app.modulos.clasificador.network_segments import SegmentGATLSTM
    from app.modulos.clasificador.pipeline import load_partition
    from app.modulos.clasificador.training import _tensor_batch, masked_classification_loss

    if args.steps < 1 or args.repeats < 1 or any(b < 1 for b in args.batch_sizes):
        raise ValueError("Pasos, repeticiones y lotes deben ser positivos")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    config = replace(Config(**dataset["config"]), device=args.device)
    device = training_device(config.device)
    preprocessor = json.loads((Path(dataset["sequence_dir"]) / "preprocessor.json").read_text())
    kinds = [preprocessor["entities"][e]["kind"] for e in preprocessor["entity_ids"]]
    partition = load_partition(dataset, "train_fit")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    results = []
    try:
        torch.manual_seed(config.seed)
        initial = SegmentGATLSTM(partition[0]["X"].shape[-1], kinds, config).to(device)
        order = torch.randperm(len(partition), generator=torch.Generator().manual_seed(47)).tolist()
        for size in args.batch_sizes:
            groups = [order[i:i + size] for i in range(0, min(len(order), size * args.steps), size)]
            groups = [g for g in groups if len(g) == size]
            if not groups:
                raise ValueError("El lote supera el número de ejemplos disponibles")
            first = partition.read_batch(groups[0])
            tensors = _tensor_batch(first, device)
            x, states, present, _, _, adjacency = tensors
            initial.eval(); initial.attention_backend = "pyg"
            with torch.no_grad():
                reference = initial(x, states, adjacency, present)
                initial.attention_backend = "dense"
                actual = initial(x, states, adjacency, present)
                max_error = max((a-b).abs().max().item() for a,b in zip(actual, reference))
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-4)
            for backend in ("pyg", "dense"):
                model = deepcopy(initial).train()
                model.attention_backend = backend
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

                def step(batch):
                    x, states, present, y, mask, adjacency = _tensor_batch(batch, device)
                    logits, types = model(x, states, adjacency, present)
                    loss = masked_classification_loss(logits, y, mask)
                    # Exercise both trained heads, backward, clipping and Adam.
                    loss = loss + F.cross_entropy(types.reshape(-1, types.shape[-1]),
                                                   y.clamp_min(0).flatten())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    return loss.detach()

                torch.manual_seed(config.seed)
                for _ in range(3):
                    step(first)
                synchronize()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                durations = []
                for _ in range(args.repeats):
                    synchronize(); started = perf_counter()
                    for group in groups:
                        loss = step(partition.read_batch(group))
                    synchronize()
                    durations.append(perf_counter() - started)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Pérdida no finita en benchmark")
                seconds = statistics.median(durations)
                result = dict(backend=backend, batch_size=size, steps=len(groups),
                              repeats=args.repeats, seconds=seconds,
                              samples_per_second=size * len(groups) / seconds,
                              max_logit_error=max_error,
                              peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                                                  if device.type == "cuda" else None))
                results.append(result)
                print(json.dumps(result), flush=True)
                del model, optimizer
    finally:
        partition.close()
    report = dict(scope="Short training-step benchmark, including reads/transfers; not a full epoch or convergence test",
                  dataset_id=dataset["dataset_id"], partition="train_fit",
                  runtime=device_metadata(device), results=results)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="JSON del dataset preparado")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
