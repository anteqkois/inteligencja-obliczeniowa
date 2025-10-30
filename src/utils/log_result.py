import pandas as pd

def log_result(algorithm_name, instance_name, params, cost, runtime):
    df = pd.DataFrame([{
        "algorithm": algorithm_name,
        "instance": instance_name,
        **params,
        "cost": cost,
        "runtime": runtime,
    }])
    df.to_csv("results/results_summary.csv", mode="a", header=False, index=False)
