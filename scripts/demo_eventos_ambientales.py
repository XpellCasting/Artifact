"""Generate and plot an explicitly artificial mixed event; never loads a model."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.modulos.clasificador.scenarios import generate_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Usa un directorio nuevo para conservar los resultados anteriores")
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    config = json.loads((ROOT / "configs/eventos_ambientales_mixtos_ejemplo.json").read_text())
    config.update(seed=17, context_seconds=259200,
                  partitions={"train_fit": [start.isoformat(), (start+timedelta(days=25)).isoformat()]},
                  calibration_end=(start+timedelta(days=14)).isoformat(),
                  quotas={"train_fit": {"episodes_per_type": 0}}, durations={})
    prep = {"entities": {"demo.temperature": {"kind": "continuous", "scale": 1},
                         "demo.motion": {"kind": "binary", "vocab": {"off": 3, "on": 4}},
                         "demo.occupancy": {"kind": "multistate", "vocab": {"empty": 3, "low": 4, "high": 5}}}}
    baseline = {"demo.temperature": "20", "demo.motion": "off", "demo.occupancy": "empty"}
    grouped = {e: [] for e in baseline}
    for step in range(25*24*12):
        for index, entity in enumerate(grouped):
            grouped[entity].append({"row_id": step*3+index, "entity": entity,
                "t": (start+timedelta(minutes=5*step)).isoformat(), "state": baseline[entity],
                "eligible": True, "certainty": "artificial_demo"})
    result = generate_scenario(None, "train_fit", "demo", config, prep, grouped=grouped)
    assert len(result["episodes"]) == 2 and not result["deficits"]
    args.output.mkdir(parents=True)
    (args.output / "escenario.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    episode = result["episodes"][0]
    begin, end = map(datetime.fromisoformat, (episode["start"], episode["end"]))
    overlay = {r["row_id"]: r for r in result["overlays"]}
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, entity in zip(axes, grouped):
        rows = [r for r in grouped[entity] if begin-timedelta(minutes=30) <= datetime.fromisoformat(r["t"]) <= end+timedelta(minutes=30)]
        ts = [datetime.fromisoformat(r["t"]) for r in rows]
        values = [overlay.get(r["row_id"], {}).get("after", r["state"]) for r in rows]
        kind = prep["entities"][entity]["kind"]
        if kind == "continuous":
            ax.plot(ts, list(map(float, values)), label="Con intervención", color="#b44d30")
            ax.plot(ts, [float(r["state"]) for r in rows], "--", label="Base artificial", color="#326b9b")
        else:
            states = list(prep["entities"][entity]["vocab"])
            ax.step(ts, [states.index(v) for v in values], where="post", color="#b44d30")
            ax.step(ts, [states.index(r["state"]) for r in rows], where="post", linestyle="--", color="#326b9b")
            ax.set_yticks(range(len(states)), states)
        ax.axvspan(begin, end, alpha=.08, color="orange")
        ax.set_ylabel(entity.replace("demo.", ""))
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Hora UTC · " + begin.strftime("%Y-%m-%d"))
    fig.suptitle("Episodio ambiental mixto · demostración artificial\nContinuo, binario y multiestado con respuestas relacionadas")
    fig.tight_layout()
    fig.savefig(args.output / "evento_mixto.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"episodes": len(result["episodes"]), "changed_rows": len(result["overlays"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
