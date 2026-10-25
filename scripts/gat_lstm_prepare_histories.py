"""Prepara los dos históricos independientes con cortes explícitos y cobertura estricta."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.modulos.clasificador.config import Config
from app.modulos.clasificador.history_preparation import plan_histories, prepare_histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--prepare-device", default="cpu")
    parser.add_argument("--prepare-batch-size", type=int, default=16)
    parser.add_argument("--normal-stride-hours", type=float, default=6)
    parser.add_argument("--episode-samples", type=int, default=12)
    parser.add_argument("--max-storage-gib", type=float, default=4)
    parser.add_argument("--reserve-free-gib", type=float, default=5)
    parser.add_argument("--probe", type=int, help="Solo prueba técnica; no publica un dataset entrenable")
    args = parser.parse_args()
    try:
        config = Config.from_json(args.config)
        Config(device=args.prepare_device)
        if min(args.workers, args.prepare_batch_size) < 1:
            raise ValueError("Procesos y lote deben ser positivos")
        if not 0 < args.max_storage_gib < float('inf') or not 0 <= args.reserve_free_gib < float('inf'):
            raise ValueError("Presupuesto o reserva inválidos")
        os.environ.update(GAT_PREPARE_WORKERS=str(args.workers), GAT_PREPARE_DEVICE=args.prepare_device,
            GAT_PREPARE_BATCH_SIZE=str(args.prepare_batch_size),
            GAT_STORAGE_MAX_BYTES=str(int(args.max_storage_gib * 1024**3)),
            GAT_STORAGE_RESERVE_BYTES=str(int(args.reserve_free_gib * 1024**3)))
        policy = {"normal_stride_seconds": args.normal_stride_hours * 3600,
                  "episode_samples": args.episode_samples}
        if args.audit_only:
            _, result = plan_histories(args.manifest, args.output, **policy)
        else:
            if (args.output / "prepared.json").exists():
                raise ValueError("Este directorio ya contiene secuencias; utiliza otro --output para repetir")
            result = prepare_histories(args.manifest, args.output, config,
                lambda message: print(message, flush=True), max_targets=args.probe, **policy)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
