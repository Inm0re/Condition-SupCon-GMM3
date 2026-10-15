# Verification of the lifecycle correction

- Four data-free regression tests cover record-level rather than window-level
  calibration, condition/record grouping, incomplete-record rejection and frozen
  calibration when scoring test records. Run `python -m unittest discover -s tests -v`.
- Synthetic aggregation and calibration were compared directly with functions
  from the archived V31 evaluator: record q90 values, median/IQR parameters,
  calibrated scores and thresholds matched exactly.
- Reapplying the corrected calibration to archived record scores matched all
  decisions in five seeds for both bearings (10 bearing-runs). Mean F1 remained
  0.8721060493 and 0.9804463365.
- An existing seed-35001 encoder checkpoint was used without encoder retraining.
  Gaussian references were reconstructed from its original normal fitting data.
  On common embeddings, public and archived component scoring agreed within
  1e-6 tolerance. All 284 full-lifecycle record decisions matched the archived
  run. Test counts were TN/FP/FN/TP = 3/13/0/45 and 6/4/0/115.
- Recomputed calibrated scores were not bitwise identical to archived scores;
  the maximum relative difference in this check was approximately 0.47%, using
  `max(abs(archived_score), 1e-5)` as the denominator. This check establishes
  matching decisions and metrics, not bitwise reproducibility across numerical
  environments. Original results were not overwritten.
- The complete lifecycle CLI passed a two-step CPU smoke run on synthetic data.
  The primary-cohort pipeline also passed its two-step CPU smoke run after the
  raw-scoring interface refactor. Old lifecycle calibration artifacts and the
  generic single-window decision interface were rejected as intended.

These checks do not constitute full retraining or independent-data validation.
No archived inputs, checkpoints, fitted parameters or prediction arrays are
distributed. The release retains the original 256-window lifecycle embedding
batch size.
