"""Event-backed state and change views; held states never become new records."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(eq=False)
class ViewProfile:
    known: pd.Series
    records: pd.Series
    fresh: pd.Series
    age: pd.Series
    changes: pd.Series
    activity: pd.Series
    negative_changes: int


def reconstruct(group, index, seconds, kind, max_age):
    group = group.sort_values('timestamp_dt', kind='stable').copy()
    buckets = group.timestamp_dt.dt.floor(f'{seconds}s')
    valid = group.is_valid_value.astype(bool)
    if kind == 'continuous':
        values = pd.to_numeric(group.get('value', group.state), errors='coerce')
        valid = valid & np.isfinite(values)
    elif kind == 'binary':
        values = pd.to_numeric(group.get('value', group.state), errors='coerce')
        fallback = group.state.astype(str).str.lower().isin({'on','open','home','active','detected','true','1'}).astype(float)
        values = values.where(values.notna(), fallback)
    else:
        values = group.state.astype(str)
    # Last event may be invalid. Its validity propagates until a new valid event.
    known = valid.groupby(buckets).last().reindex(index).ffill().fillna(False).astype(bool)
    levels = values.where(valid).groupby(buckets).last().reindex(index).ffill().where(known)
    records = group.groupby(buckets).size().reindex(index, fill_value=0).gt(0)
    latest = group.timestamp_dt.groupby(buckets).last().reindex(index).ffill()
    ages = ((index.to_series() + pd.Timedelta(seconds=seconds)) - latest).dt.total_seconds()
    fresh = known & ages.le(max_age)
    previous_valid = valid.shift(1, fill_value=False)
    if kind == 'continuous':
        delta = values.diff().where(valid & previous_valid)
        activity = delta.ne(0) & delta.notna()
        negative = int(delta.lt(0).sum())
    else:
        activity = values.ne(values.shift(1)) & valid & previous_valid
        delta = activity.astype(float).where(valid & previous_valid)
        negative = 0
    changes = delta.groupby(buckets).sum(min_count=1).reindex(index)
    predecessor_known = known.shift(1, fill_value=False)
    changes = changes.where(records, 0.).where(known & (predecessor_known | delta.notna().groupby(buckets).any().reindex(index, fill_value=False)))
    changes = changes.where(valid.groupby(buckets).all().reindex(index, fill_value=True))
    active = activity.groupby(buckets).any().reindex(index, fill_value=False)
    profile = ViewProfile(known, records, fresh, ages, changes, active, negative)
    levels.attrs['profile'] = profile
    levels.attrs['relation_view'] = 'state'
    return levels, profile


def change_view(signal):
    profile = signal.attrs.get('profile')
    if profile is None:
        return signal.diff()
    result = profile.changes.copy()
    result.attrs = {'profile': profile, 'relation_view': 'changes'}
    return result


def pair_support(source, target, lag, view, total_windows):
    left, right = source.attrs.get('profile'), target.attrs.get('profile')
    if left is None or right is None:
        return source, target.shift(lag), None
    target_aligned = target.shift(lag)
    known_left = source.notna()
    known_right = target_aligned.notna()
    paired = known_left & known_right
    observable = known_left | known_right
    activity = left.activity | right.activity.shift(lag, fill_value=False)
    records = left.records | right.records.shift(lag, fill_value=False)
    sampling = records if view == 'state' else activity | activity.shift(1, fill_value=False)
    usable = paired & sampling
    basis = records if view == 'state' else activity
    independent = paired & basis
    denominator = int(observable.sum()) if view == 'state' else int(basis.sum())
    numerator = int(paired.sum()) if view == 'state' else int(independent.sum())
    fresh = left.fresh & right.fresh.shift(lag, fill_value=False)
    report_pair = left.records & right.records.shift(lag, fill_value=False)
    left_start = left.records[left.records].index.min()
    right_records = right.records.shift(lag, fill_value=False)
    right_start = right_records[right_records].index.min()
    shared = source.index >= max(left_start, right_start) if pd.notna(left_start) and pd.notna(right_start) else np.zeros(len(source), dtype=bool)
    if view == 'state':
        denominator = int(np.sum(shared))
        numerator = int((paired & shared).sum())
    else:
        denominator = int((basis & shared).sum())
        numerator = int((independent & shared).sum())
    days = int(source.index[independent].normalize().nunique())
    metadata = {
        'basis': 'known_state_overlap' if view == 'state' else 'comparable_change_events',
        'coverage_ratio': numerator / denominator if denominator else 0.,
        'known_time_coverage': float(paired.mean()) if len(paired) else 0.,
        'coincident_record_coverage': int(report_pair.sum()) / total_windows if total_windows else 0.,
        'fresh_time_coverage': int((paired & fresh).sum()) / max(1, int(paired.sum())),
        'independent_observations': int(independent.sum()), 'support_days_utc': days,
        'sample_windows': int(usable.sum()), 'comparable_windows': numerator,
        'coverage_denominator': denominator,
        'coverage_scope': 'shared_entity_history',
        'source_negative_changes': left.negative_changes, 'target_negative_changes': right.negative_changes,
        'source_max_age_seconds': float(left.age[usable].max()) if usable.any() else None,
        'target_max_age_seconds': float(right.age.shift(lag)[usable].max()) if usable.any() else None,
    }
    return source.where(usable), target_aligned.where(usable), metadata


def observed_changes(events, reference, seconds, kind):
    """Window (reference-seconds, reference], seeded by earlier events."""
    if events.empty or pd.isna(reference):
        return None, {'missing_reason': 'no_previous_observation'}
    end = int(events.index.searchsorted(reference, side='right'))
    if not end:
        return None, {'missing_reason': 'no_previous_observation'}
    available = events.iloc[:end]
    if not bool(available.iloc[-1].is_valid_value):
        return None, {'missing_reason': 'invalid_latest_observation'}
    start = reference - pd.Timedelta(seconds=seconds)
    first = int(available.index.searchsorted(start, side='right'))
    if not first:
        return None, {'missing_reason': 'no_state_before_change_window'}
    window = available.iloc[first-1:]
    valid = window.is_valid_value.astype(bool)
    if not valid.all():
        return None, {'missing_reason': 'invalid_event_in_change_window'}
    if kind == 'continuous':
        values = pd.to_numeric(window.get('value', window.state), errors='coerce')
        if not np.isfinite(values).all():
            return None, {'missing_reason': 'invalid_numeric_change'}
        value = float(values.diff().iloc[1:].sum())
    else:
        value = float(window.state.astype(str).ne(window.state.astype(str).shift(1)).iloc[1:].sum())
    return value, {'observation_method': 'recorded_changes', 'observed_timestamp': available.index[-1].isoformat(),
                   'observation_age_seconds': (reference-available.index[-1]).total_seconds(),
                   'records_in_window': end-first, 'carried_forward': end == first, 'missing_reason': None}
