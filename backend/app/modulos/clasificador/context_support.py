"""Conservative support from disjoint original 72h windows, never replicas."""
from datetime import timedelta
from app.modulos.clasificador.config import parse_utc


def disjoint_context_count(rows):
    intervals=[]
    for row in rows:
        end=row.get("context_end") or row.get("target_time")
        if end is None: continue
        end=parse_utc(end)
        start=parse_utc(row["context_start"]) if row.get("context_start") else end-timedelta(hours=72)
        intervals.append((end,start))
    previous=None;count=0
    for end,start in sorted(set(intervals)):
        if previous is None or start>previous:
            count+=1;previous=end
    return count
