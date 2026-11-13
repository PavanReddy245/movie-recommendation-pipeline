import argparse, json
from pipeline_tasks import TASKS
p=argparse.ArgumentParser(); p.add_argument('stage',choices=TASKS); args=p.parse_args()
print(json.dumps(TASKS[args.stage](),indent=2,default=str))
