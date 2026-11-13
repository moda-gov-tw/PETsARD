# run_json2.py -- JSON demo 2 of 3: SCALE TEST (public dataset, full size)
#
# What this one is for: proving the JSON loader holds up on a large public
# dataset file rather than a hand-built toy, and that the result matches the CSV original.

from petsard import Executor

exec = Executor(config='demo_json/configs/adult-income-json.yaml')
exec.run()
