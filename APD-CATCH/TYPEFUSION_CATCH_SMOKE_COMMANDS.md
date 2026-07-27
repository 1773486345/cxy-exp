/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m compileall ts_benchmark/baselines/typefusion_catch

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m ts_benchmark.baselines.typefusion_catch.smoke

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m unittest discover -s ts_benchmark/baselines/typefusion_catch/tests -p 'test_*.py'

git diff --check

git diff -- ts_benchmark/baselines/catch/
