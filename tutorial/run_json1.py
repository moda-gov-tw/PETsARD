# run_json1.py -- JSON demo 1 of 3: FEATURE TEST (smallest, fastest)
#
# What this one is for: proving the JSON reader and the JSON output of the
# Reporter work at all, on data small enough that a failure is obvious by eye.
# The synthesizer is the default method (PETsARD Gaussian Copula).

from petsard import Executor

exec = Executor(config='demo_json/configs/test_json.yaml')
exec.run()
