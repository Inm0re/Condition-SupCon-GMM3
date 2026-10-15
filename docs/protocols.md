# Protocols and interpretation

Normal-only refers to health state: no fault samples enter encoder fitting,
density estimation or normal calibration. Operating-condition metadata supervise
representation learning and route scoring. GMM3 is a hard minimum of prior-adjusted
Gaussian component scores, not a weighted soft-mixture likelihood.

HUST: length 2048, stride 256. Normal fitting/calibration positions per condition
are 0–396 / 404–504 of the first-half normal window pool; starts 397–403 are
discarded. The normal test block is the second half. Healthy source files are
shared chronologically; this is not source-record independence.

Paderborn: K001–K004 fitting, K005 calibration, K006 normal testing, with the 14
real-damaged bearings as fault testing. Two current channels and one vibration
channel are retained. The uniform condition/repetition exclusion and 122 windows
per retained file are encoded in the preparation script. Artificially damaged
bearings are not part of the REAL evaluation.

SEU: seven vibration channels; health_20_0 and health_30_2. Positions 0–203 fit,
204–254 calibrate, 255 is excluded, 256–510 are normal test. Source identity is
retained in the generated file metadata. The same healthy sources cross temporal
partitions; source files are not automatically physical bearing identities.

XJTU: Bearing1-1 and Bearing2-2, 16 non-overlapping windows per minute record.
Normal prefixes use chronological 60/20/20 splitting. The operational labelled
fault boundaries are records 79 and 47. These are not independently measured
physical degradation onsets. The archived V48 runner calls the V31 evaluator:
q90 aggregation of 16 raw window scores is performed on BOTH normal calibration
records and test records. Each bearing's median and IQR are estimated from its
normal calibration-record q90 scores, with IQR floored at 1e-8. The q95 of these
calibrated normal record scores fixes that bearing's threshold. Test records
use the same frozen median, IQR and threshold. This is the policy underlying the
reported five-seed lifecycle results; it does not use fault labels to calibrate.

Release v1.0.0 incorrectly applied a calibration-window q95 to test-record q90
scores, omitted median–IQR scaling and described that as archived behavior.
That packaging error is corrected in v1.0.1; the original experiment and paper
results are not replaced. The old tag is retained for an auditable history.

K=3 is an empirical shared configuration, not selected by a normal-only criterion.
Five algorithmic seeds are not five physical replications. The auxiliary random
permutation retains the documented same-condition/self-pair ambiguities.

Public-only macro summaries cover only the public datasets present in your run.
They cannot reproduce the paper's four-cohort macro including restricted data.
Do not mix original archived baseline calibration with the revised common-C
comparison. No training data, split arrays, checkpoints, density banks or
calibration parameters are distributed in this repository.
