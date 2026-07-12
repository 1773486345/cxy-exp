# Direction B

B1 is the retained executable multi-evidence repair baseline. The B2/B3
subdirectories contain closed transfer hypotheses and their retained records;
they are not active experiment entry points.

| Area | Location |
| --- | --- |
| B result interpretation | [`B_RESULT_INDEX.md`](B_RESULT_INDEX.md) |
| B1 protocol | [`B1_EXPERIMENT_PLAN.md`](B1_EXPERIMENT_PLAN.md) |
| Closed B2/B3 records | [`closed/`](closed/) |
| Runtime code and tools | [`../../scripts/multi_evidence/`](../../scripts/multi_evidence/README.md) |
| Model implementation | `../../ts_benchmark/baselines/MultiEvidenceRepair/` |
| Result artifacts | `../../result/multi_evidence/` |

Direction B code, score components, calibration, and thresholds must not be
used as A2 internals outside an A2-specific frozen protocol.
