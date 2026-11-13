# run_json3.py -- JSON demo 3 of 3: FULL EVALUATION SUITE
#
# What this one is for: proving the evaluation side works on JSON-sourced data.
# It is the only one of the three with a Splitter, and it runs five evaluators
# covering all three assessment families.

from petsard import Executor

exec = Executor(config='demo_json/configs/customer-churn-json.yaml')
exec.run()
