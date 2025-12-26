import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
if __name__ == "__main__":
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from src.utils.tsp_loader import load_tsp_matrix
    from src.utils.distance import route_length_fast

    # ## 1. Load New Results

    results_path = os.path.join(
        current_dir, "results_new/SA/2025-12-25__10-31__3942_sec__results.csv"
    )
    if not os.path.exists(results_path):
        print(f"File not found: {results_path}")
    else:
        df_results = pd.read_csv(results_path)
        print(f"Loaded {len(df_results)} rows.")
        print(df_results.head())

    # ## 2. Verify Costs
    # We iterate through unique instances found in the results, load the corresponding matrix, and check a sample of routes.

    unique_instances = df_results["instance"].unique()

    print(
        f"{'Instance':<20} | {'Reported':<10} | {'Calculated':<10} | {'Diff':<10} | {'Valid?':<6}"
    )
    print("-" * 70)

    valid_count = 0
    total_checked = 0

    for instance in unique_instances:
        # Load matrix for this instance
        try:
            matrix = load_tsp_matrix(instance)
        except FileNotFoundError:
            print(f"Could not load data for {instance}, skipping.")
            continue

        # Get subset of results for this instance
        df_inst = df_results[df_results["instance"] == instance]

        # Check all or a sample. Let's check first 5 and last 5 to be sure.
        sample_indices = list(range(min(5, len(df_inst))))
        if len(df_inst) > 5:
            sample_indices += list(range(len(df_inst) - 5, len(df_inst)))

        # Making sample unique just in case
        sample_indices = sorted(list(set(sample_indices)))

        for i in sample_indices:
            row = df_inst.iloc[i]
            reported_cost = row["min_cost"]
            route_str = row["min_route"]

            try:
                route = np.array([int(x) for x in route_str.split("-")], dtype=np.int32)
            except ValueError:
                print(f"{instance:<20} | Error parsing route: {route_str}")
                continue

            # Check if route indices match matrix size
            if len(route) != matrix.shape[0]:
                print(
                    f"{instance:<20} | Size mismatch: Route {len(route)} vs Matrix {matrix.shape[0]}"
                )
                continue

            # calc_cost = route_length_fast(matrix, route)
            calc_cost = round(route_length_fast(matrix, route), 3)

            diff = abs(reported_cost - calc_cost)
            # Relax tolerance to 1e-2 to account for CSV precision loss (2 decimal places)
            is_valid = diff < 1e-2

            valid_str = "OK" if is_valid else "FAIL"
            # Print diff with more precision to see the actual error
            print(
                f"{instance:<20} | {reported_cost:<10.2f} | {calc_cost:<10.2f} | {diff:<10.5f} | {valid_str}"
            )

            total_checked += 1
            if is_valid:
                valid_count += 1

    print("-" * 70)
    print(f"Total Checked: {total_checked}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {total_checked - valid_count}")
