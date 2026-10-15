"""Record-level q90 aggregation and normal-record calibration used by V31/V48."""
import numpy as np


def aggregate_records(window_scores, conditions, record_ids, windows_per_record=16):
    """Group by (condition, record), retaining the archived float32 record scale."""
    raw = np.asarray(window_scores, dtype=float)
    conditions = np.asarray(conditions)
    record_ids = np.asarray(record_ids)
    if raw.ndim != 1 or conditions.shape != raw.shape or record_ids.shape != raw.shape:
        raise ValueError('Scores, conditions and record IDs must be matching 1D arrays')
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Record aggregation requires nonempty finite window scores')
    values, groups, records = [], [], []
    for condition in np.unique(conditions):
        rows = np.flatnonzero(conditions == condition)
        for record in np.unique(record_ids[rows]):
            selected = rows[record_ids[rows] == record]
            if len(selected) != windows_per_record:
                raise ValueError(f'Condition {condition}, record {record}: expected {windows_per_record} windows, got {len(selected)}')
            values.append(np.quantile(raw[selected], .90))
            groups.append(int(condition))
            records.append(int(record))
    return dict(raw=np.asarray(values, np.float32), condition=np.asarray(groups, np.int64), record=np.asarray(records, np.int64))


def fit_record_calibration(records):
    """Fit median/IQR and q95 on NORMAL calibration-record q90 scores only."""
    scales, thresholds = {}, {}
    conditions = records['condition']
    for condition in np.unique(conditions):
        local = np.asarray(records['raw'][conditions == condition], np.float32)
        median = float(np.median(local))
        q1, q3 = np.quantile(local, (.25, .75))
        iqr = max(float(q3 - q1), 1e-8)
        calibrated = (local - median) / iqr
        key = str(int(condition))
        thresholds[key] = float(np.quantile(calibrated, .95))
        scales[key] = dict(median=median, iqr=iqr, calibration_records=int(len(local)))
    return scales, thresholds


def apply_record_calibration(records, scales, thresholds):
    score = np.empty(len(records['raw']), np.float32)
    threshold = np.empty(len(score), np.float32)
    for condition in np.unique(records['condition']):
        key = str(int(condition))
        if key not in scales or key not in thresholds:
            raise ValueError(f'Unsupported lifecycle condition: {condition}')
        rows = records['condition'] == condition
        pars = scales[key]
        score[rows] = (records['raw'][rows] - pars['median']) / pars['iqr']
        threshold[rows] = thresholds[key]
    return score, (score >= threshold).astype(np.int8), threshold
