# MSL Boundary × Scoring-Object Comparison

This comparison changes window boundary treatment only in the new Full run and compares two score aggregations from the same 55-field fitted GMM. All 55 fields remain model inputs; `col_0`-only changes only final scoring.

- Old run: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/score/MNRF-ViewProbe/MSL/full_gmm-full-20261021T112147Z-2470335`
- New run: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/score/MNRF-ViewProbe-MSL-Boundary/MSL/full_gmm-sequence-bounded-20261102T105427Z-2415811`
- Common valid test endpoints: 68572 / official test points 132046.
- Four cells: old concatenated × 55-variable max; old concatenated × col_0; new sequence-bounded × 55-variable max; new sequence-bounded × col_0.
- Every cell derives its own validation median/MAD and calibration q=0.995 threshold; prediction is strictly score > threshold.
- The common endpoint intersection is used without filling missing scores. Event metrics respect the 27 mapped test-channel boundaries.
- Old 0.159/0.279 figures are not inserted; all rows are recomputed from the actual saved per-variable NLL on the common endpoints.
- The new result does not exist until the manual launcher completes; this tool itself only reads and validates saved artifacts.
