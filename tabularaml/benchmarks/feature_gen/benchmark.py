from tabularaml.utils.datasets import Dataset
from tabularaml.generate.features import FeatureGenerator
from tqdm.auto import tqdm
import os
import pandas as pd
import numpy as np

class FeatureGeneratorBenchmark:
    """
    A helper class to benchmark the automatic feature generator.
    """

    def __init__(self, 
                 mode = "best",
                 generator: FeatureGenerator = None,
                 rounds = 100,
                 tasks = ["regression", "classification"],
                 timeout = None,
                 save_as = "feature_gen",
                 save_freq = 5):
        self.mode = mode
        self.generator = generator if generator is not None else \
                         FeatureGenerator(mode = mode, 
                                          time_budget = timeout // rounds if timeout is not None else None,
                                          log_to_file = None)
        self.rounds = rounds
        self.tasks = tasks
        self.timeout = timeout
        self.mode = mode if mode is not None else self.generator.mode
        self.save_as = r"tabularaml\benchmarks\feature_gen" + fr"\{save_as}_{self.mode}.csv" 
        self.results = None
        self.save_freq = save_freq  # How often to save results (every N datasets)

    

    def run(self):

        self.results = {
            "tid": [], "task_type": [], "elapsed_time": [], "n_samples": [], "n_init_feats": [],
            "n_added_feats": [], "metric": [], "initial_metric": [], "final_metric": [],
            "gain": [], "pct_gain": []
        }
        
        print("STARTING THE BENCHMARK:")
        print(f"RUNNING FOR {self.rounds} ROUNDS, WITH TIMEOUT: {self.timeout}.")
        print(f"SAVING RESULTS EVERY {self.save_freq} DATASET(S)")

        # Check if results file already exists and load it if it does
        existing_results = None
        if os.path.exists(self.save_as):
            existing_results = pd.read_csv(self.save_as)
            print(f"Found existing results file with {len(existing_results)} entries")
        
        for i in tqdm(range(self.rounds)):
            
            print(f"*\tDataset {i}:")
            
            while True:
                try:
                    dataset = Dataset.fetch_random_tasks(task=self.tasks, k=1)[0]
                    X, y = dataset.X, dataset.y
                    self.generator.task = dataset.task
                    res = self.generator.generate(X, y)
                except Exception as e:
                    print(f"{type(e).__name__} occurred while fetching dataset: {str(e)[:100]}. Trying another...")
                else:
                    break
                    
            self.results["tid"].append(dataset.id)
            self.results["task_type"].append(dataset.task)
            self.results["elapsed_time"].append(self.generator.elapsed_time)
            self.results["n_samples"].append(self.generator.n_samples)
            self.results["n_init_feats"].append(self.generator.n_init_feats)
            self.results["n_added_feats"].append(self.generator.n_added_feats)
            self.results["metric"].append("rmse" if dataset.task == "regression" else \
                                         "binary_crossentropy" if len(np.unique(y)) == 2 \
                                          else "categorical_crossentropy")
            self.results["initial_metric"].append(self.generator.initial_metric)
            self.results["final_metric"].append(self.generator.final_metric)
            self.results["gain"].append(self.generator.gain)
            self.results["pct_gain"].append(100 * self.generator.pct_gain)
            
            # Save results based on save_freq parameter
            if (i + 1) % self.save_freq == 0 or i == self.rounds - 1:  # Save every save_freq datasets or at the end
                # Save used ids
                Dataset.save_used_ids()
                # Create a dataframe from all new results since last save
                start_idx = max(0, len(self.results["tid"]) - self.save_freq)
                current_results = {key: values[start_idx:] for key, values in self.results.items()}
                current_results_df = pd.DataFrame(current_results)
                if existing_results is not None:
                    # Append new results to existing results
                    updated_results = pd.concat([existing_results, current_results_df], axis=0)
                    updated_results.to_csv(self.save_as, index=False)
                    existing_results = updated_results
                    print(f"Saved {len(current_results_df)} new results (total: {len(updated_results)})")
                else:
                    # Create new results file
                    current_results_df.to_csv(self.save_as, index=False)
                    existing_results = current_results_df
                    print(f"Created results file with {len(current_results_df)} entries")
                
            print("-" * 150, "\n\n\n")
            




