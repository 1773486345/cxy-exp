/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m compileall ts_benchmark/baselines/typefusion_catch

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m ts_benchmark.baselines.typefusion_catch.smoke

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m unittest ts_benchmark.baselines.typefusion_catch.tests.test_three_stage_state_continuity

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m unittest ts_benchmark.baselines.typefusion_catch.tests.test_equal_total_training_budget

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m unittest ts_benchmark.baselines.typefusion_catch.tests.test_full_model_causal_no_leakage

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m unittest ts_benchmark.baselines.typefusion_catch.tests.test_full_model_relation_mask_no_leakage

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m unittest ts_benchmark.baselines.typefusion_catch.tests.test_branch_pretrain_skips_joint_modules

/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python -m unittest discover -s ts_benchmark/baselines/typefusion_catch/tests -p 'test_*.py'

git diff --check

git diff -- ts_benchmark/baselines/catch/
