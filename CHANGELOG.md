# Release notes

## v1.0.1 — XJTU-SY record-calibration correction

The initial public lifecycle runner did not reproduce the archived evaluation
used by the manuscript. It calculated q95 from normal calibration **windows**,
set the alignment parameters to median=0/IQR=1, and compared test-record q90
scores with that window threshold. The v1.0.0 documentation also incorrectly
identified that behavior as the archived policy.

The actual five-seed experiment (`xjtu_h_gmm3_5seed_v48.py`) calls
`eval_v31.evaluate`: aggregate both calibration and test raw window scores to
record q90 first, estimate median/IQR from normal calibration records, and take
q95 of the calibrated normal record scores. v1.0.1 restores this order in the
public lifecycle runner. The new `lifecycle_scoring` module exposes this
record-level evaluation independently of encoder training.

A separate `lifecycle_infer` entry point requires record IDs and complete groups
of 16 windows. Generic window inference rejects lifecycle artifacts. The
corrected entry point rejects calibration bundles created by the initial
lifecycle runner, rather than silently treating them as corrected artifacts.

This corrects the public packaging, not the archived experiment. It does not
change the paper's labels, normal partitions, thresholds or reported results,
and it does not introduce a test-driven threshold adjustment. Primary-cohort
calibration and scoring remain unchanged. The v1.0.0 tag is not rewritten.

## v1.0.0

Initial code-only release; its XJTU-SY calibration implementation is superseded
by v1.0.1. No datasets, trained weights or fitted parameters are distributed.
