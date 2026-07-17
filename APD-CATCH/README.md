# CATCH

This repository retains the official CATCH implementation and its benchmark
infrastructure.

- [`ts_benchmark/baselines/catch/`](./ts_benchmark/baselines/catch/) is the
  usable original CATCH baseline.
- [`ts_benchmark/baselines/apd_catch/`](./ts_benchmark/baselines/apd_catch/)
  is frozen legacy exploratory code. Its existing results remain historical
  records and are not the active comparison path.
- Subsequent work will implement new model baselines directly and compare them
  on real data against original CATCH.

## Original CATCH

Install the repository dependencies, prepare the TAB data under `dataset/`,
then use the original score or label scripts, for example:

```bash
pip install -r requirements.txt
sh ./scripts/multivariate_detection/detect_label/MSL_script/CATCH.sh
sh ./scripts/multivariate_detection/detect_score/MSL_script/CATCH.sh
```

The upstream CATCH model is implemented in
[`ts_benchmark/baselines/catch/CATCH.py`](./ts_benchmark/baselines/catch/CATCH.py).
