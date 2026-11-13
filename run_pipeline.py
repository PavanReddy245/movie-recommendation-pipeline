"""Run all pipeline stages sequentially with explicit dependencies."""
from pipeline_tasks import TASKS

def main():
    stages=['ingest','validate','prepare','transform','features','train','evaluate','track']
    for i,name in enumerate(stages,1):
        print(f'[{i}/8] {name.title()}')
        result=TASKS[name]()
        print(result)
    print('Pipeline completed successfully.')

if __name__=='__main__': main()
