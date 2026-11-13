# run_json1.py -- JSON demo 1 of 3: FEATURE TEST (smallest, fastest)
#
# What this one is for: proving the three JSON-related code paths work at all,
# on data small enough that a failure is obvious by eye.

from petsard import Executor

exec = Executor(config='demo_json/configs/test_json_advgan.yaml')
exec.run()
