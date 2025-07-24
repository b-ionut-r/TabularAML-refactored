import numpy as np
import pandas as pd
import random
import openml
from openml.tasks import TaskType
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch
import os




def fetch_random_openml_dataset(task_type=None, return_X_y=True, as_frame=True, verbose=True, 
                               min_samples=1000, max_samples=10000, min_features=20, max_features=100,
                               exclude_ids=None):
    """
    Fetch a random dataset from OpenML with proper task type handling.
    
    Parameters:
    -----------
    task_type : str or None
        Type of task. Can be 'regression', 'classification', or None (for both).
    return_X_y : bool, default=True
        If True, returns (data, target) instead of a Bunch object.
    as_frame : bool, default=True
        If True, the data is a pandas DataFrame.
    verbose : bool, default=True
        Print information about the dataset.
    min_samples : int, default=1000
        Minimum number of samples in the dataset.
    max_samples : int, default=10000
        Maximum number of samples in the dataset.
    min_features : int, default=20
        Minimum number of features in the dataset.
    max_features : int, default=100
        Maximum number of features in the dataset.
    exclude_ids : list or None, default=None
        List of dataset IDs to exclude from selection.
        
    Returns:
    --------
    X, y : tuple
        Features and target if return_X_y is True, otherwise a Bunch object.
    """
    # Convert exclude_ids to a set for faster lookups, if provided
    exclude_set = set(exclude_ids) if exclude_ids else set()
    
    try:
        # 1. First get task list with proper task type
        if task_type in ['classification', 'regression']:
            # Set the appropriate TaskType
            if task_type == 'classification':
                openml_task_type = TaskType.SUPERVISED_CLASSIFICATION
            else:  # regression
                openml_task_type = TaskType.SUPERVISED_REGRESSION
        else:
            # If no specific task_type specified, choose randomly
            openml_task_type = random.choice([
                TaskType.SUPERVISED_CLASSIFICATION,
                TaskType.SUPERVISED_REGRESSION
            ])
            
        if verbose:
            task_name = "classification" if openml_task_type == TaskType.SUPERVISED_CLASSIFICATION else "regression"
            print(f"Searching for {task_name} tasks...")
            
        # Get list of tasks matching the task type
        tasks = openml.tasks.list_tasks(task_type=openml_task_type, output_format="dataframe")
        
        if verbose:
            print(f"Found {len(tasks)} {task_name} tasks")
            
        if len(tasks) == 0:
            raise ValueError(f"No tasks found for task type: {task_name}")
            
        # 2. Get the dataset IDs from these tasks
        dataset_ids = tasks['did'].unique().tolist()
        
        # Filter out excluded IDs
        if exclude_set:
            original_count = len(dataset_ids)
            dataset_ids = [did for did in dataset_ids if did not in exclude_set]
            if verbose and len(dataset_ids) < original_count:
                print(f"Filtered out {original_count - len(dataset_ids)} excluded dataset IDs")
                
        if not dataset_ids:
            raise ValueError("All available datasets have been excluded")
        
        # 3. Filter datasets by number of samples and features
        filtered_datasets = []
        for dataset_id in dataset_ids:
            try:
                # Get dataset information
                dataset_info = openml.datasets.get_dataset(dataset_id, download_data=False)
                
                # Check if dataset meets criteria
                if (min_samples <= dataset_info.qualities['NumberOfInstances'] <= max_samples and
                    min_features <= dataset_info.qualities['NumberOfFeatures'] <= max_features):
                    filtered_datasets.append(dataset_id)
                    
                    # Once we have enough datasets, stop checking to improve performance
                    if len(filtered_datasets) >= 20:
                        break
            except Exception as e:
                if verbose:
                    print(f"Could not access dataset {dataset_id}: {str(e)}")
                continue
        
        if not filtered_datasets:
            if verbose:
                print("No datasets match the criteria. Relaxing constraints...")
            # Relax constraints if no datasets found
            for dataset_id in dataset_ids[:50]:  # Check just the first 50 to save time
                try:
                    dataset_info = openml.datasets.get_dataset(dataset_id, download_data=False)
                    if dataset_info.qualities['NumberOfInstances'] <= max_samples * 2:
                        filtered_datasets.append(dataset_id)
                    
                    # Once we have enough datasets, stop checking
                    if len(filtered_datasets) >= 10:
                        break
                except Exception as e:
                    if verbose:
                        print(f"Could not access dataset {dataset_id} during relaxed search: {str(e)}")
                    continue
        
        if not filtered_datasets:
            raise ValueError("No suitable datasets found even with relaxed constraints")
            
        # Select a random dataset ID
        dataset_id = random.choice(filtered_datasets)
        
        if verbose:
            print(f"Fetching dataset with ID: {dataset_id}")
        
        # Fetch the dataset
        data = fetch_openml(data_id=dataset_id, return_X_y=return_X_y, as_frame=as_frame)
        
        # If we want to return a Bunch object and not X,y tuple
        if not return_X_y:
            if verbose:
                # Access information about the dataset
                if hasattr(data, 'details') and isinstance(data.details, dict) and 'name' in data.details:
                    print(f"Dataset name: {data.details['name']}")
                else:
                    print(f"Dataset name: Unknown")
                print(f"Number of samples: {data.data.shape[0]}")
                print(f"Number of features: {data.data.shape[1]}")
                if hasattr(data, 'target_names') and data.target_names is not None:
                    print(f"Target names: {data.target_names}")
                if hasattr(data, 'feature_names') and data.feature_names is not None and len(data.feature_names) > 0:
                    print(f"Feature names: {data.feature_names[:5]}... (showing first 5)")
        # If we're returning X,y tuple and want to show info
        elif verbose:
            X, y = data
            print(f"Dataset successfully fetched:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape if hasattr(y, 'shape') else 'scalar'}")
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                unique_values = y.nunique()
                print(f"Number of unique target values: {unique_values}")
                if unique_values <= 10:
                    print(f"Target value distribution: {y.value_counts().to_dict()}")
                else:
                    print(f"Target statistics: min={y.min()}, max={y.max()}, mean={y.mean()}")
            
        return data
    
    except Exception as e:
        if verbose:
            print(f"Error in primary fetch method: {str(e)}")
            print("Falling back to direct fetch method...")
        return _fetch_fallback(task_type, return_X_y, as_frame, verbose, exclude_set)

def _fetch_fallback(task_type=None, return_X_y=True, as_frame=True, verbose=True, exclude_set=None):
    """Fallback method that attempts to fetch datasets by ID."""
    # Some reliable dataset IDs as fallback
    classification_datasets = [40994, 554, 1590, 40996, 40975]  # Using integers
    regression_datasets = [42225, 531, 560, 42395, 41027]
    
    if task_type == 'classification':
        dataset_ids = classification_datasets
    elif task_type == 'regression':
        dataset_ids = regression_datasets
    else:
        dataset_ids = classification_datasets + regression_datasets
    
    # Filter out excluded IDs
    original_count = len(dataset_ids)
    if exclude_set:
        dataset_ids = [did for did in dataset_ids if did not in exclude_set]
        if verbose and len(dataset_ids) < original_count:
            print(f"Filtered out {original_count - len(dataset_ids)} excluded dataset IDs from fallback list")
            
    if not dataset_ids:
        raise ValueError("All available fallback datasets have been excluded")
    
    # Create a copy of dataset_ids to allow for multiple attempts
    remaining_ids = dataset_ids.copy()
    
    while remaining_ids:
        dataset_id = random.choice(remaining_ids)
        remaining_ids.remove(dataset_id)  # Remove to avoid trying the same ID twice
        
        try:
            if verbose:
                print(f"Fallback: fetching dataset with ID: {dataset_id}")
            
            data = fetch_openml(data_id=dataset_id, return_X_y=return_X_y, as_frame=as_frame)
            return data
            
        except Exception as e:
            if verbose:
                print(f"Fallback failed for dataset {dataset_id}: {str(e)}")
            # Continue to next dataset
    
    # If we've tried all datasets and none worked
    raise ValueError("All fallback datasets failed.")

# Example usage
if __name__ == "__main__":
    print("======= FETCH RANDOM DATASET =======")
    try:
        result = fetch_random_openml_dataset(task_type=None)
        if isinstance(result, tuple):
            X, y = result
            print(f"Success! Dataset has {X.shape[0]} samples and {X.shape[1]} features")
        else:
            print(f"Success! Dataset returned as a Bunch object")
    except Exception as e:
        print(f"Error: {str(e)}")
        
    print("\n======= FETCH CLASSIFICATION DATASET =======")
    try:
        result = fetch_random_openml_dataset(task_type='classification')
        if isinstance(result, tuple):
            X, y = result
            print(f"Success! Classification dataset has {X.shape[0]} samples and {X.shape[1]} features")
            if hasattr(y, 'nunique'):
                print(f"Number of classes: {y.nunique()}")
        else:
            print(f"Success! Classification dataset returned as a Bunch object")
    except Exception as e:
        print(f"Error: {str(e)}")
        
    print("\n======= FETCH REGRESSION DATASET =======")
    try:
        result = fetch_random_openml_dataset(task_type='regression')
        if isinstance(result, tuple):
            X, y = result
            print(f"Success! Regression dataset has {X.shape[0]} samples and {X.shape[1]} features")
            if hasattr(y, 'describe'):
                print(f"Target range: {y.min()} to {y.max()}")
        else:
            print(f"Success! Regression dataset returned as a Bunch object")
    except Exception as e:
        print(f"Error: {str(e)}")



import pandas as pd
from typing import Optional
import openml
import random
import warnings
import os

class Dataset:
    """
    Container class for fetching OPENML datasets.
    """
    BASE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
    task_ids = {
        "regression": pd.read_csv(BASE_PATH + "reg_ids.csv")["tid"].to_list(),  # with <=25000 inst
        "classification": pd.read_csv(BASE_PATH + "cls_ids.csv")["tid"].to_list()
    }
    all_ids = task_ids["regression"] + task_ids["classification"]
    used_ids = ([] if not os.path.exists(BASE_PATH + "used_ids.csv")
                else pd.read_csv(BASE_PATH + "used_ids.csv")["tid"].to_list())

    def __init__(self, X: pd.DataFrame, 
                 y: Optional[pd.Series] = None, 
                 id = None,
                 task = None):
        self.X = X
        self.y = y
        self.id = id
        self.task = task if task else self._infer_task()
    
    def _infer_task(self):
        # Infer task type
        if self.y is None:
            task = "unsupervised"
        elif self.y.dtype.kind in 'biufc':  # Numeric types
            task = "regression"
        else:
            task = "classification"
        return task

    @classmethod
    def fetch_by_id(cls, id):
        with warnings.catch_warnings():
            # Ignore all warnings
            warnings.filterwarnings("ignore")
            try:
                task = openml.tasks.get_task(id)
                type = "regression" if id in cls.task_ids["regression"] else "classification"
                X, y = task.get_X_and_y(dataset_format="dataframe")
                
                # Check for text columns
                has_text = False
                for col in X.columns:
                    col_data = X[col]
                    if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                        unique_count = col_data.nunique()
                        total = len(col_data)
                        if total == 0:
                            continue  # Avoid division by zero
                        if unique_count / total > 0.9:  # Threshold set to 90%
                            has_text = True
                            break
                
                if has_text:
                    cls.used_ids.append(id)
                    return None
                
                if id in cls.used_ids:
                    print(f"Warning: Dataset with ID {id} was already used.")
                else:
                    cls.used_ids.append(id)
                return cls(X, y, id, type)
            except Exception as e:
                cls.used_ids.append(id)
                return None
    
    @classmethod
    def fetch_random_id(cls):
        """
        Fetch a random unseen dataset, choosing task‐types uniformly,
        then sampling uniformly among that task’s unseen IDs.
        """
        # build per‐task unseen sets
        unseen_by_task = {
            t: list(set(ids) - set(cls.used_ids))
            for t, ids in cls.task_ids.items()
        }
        # filter out tasks with no unseen IDs
        available_tasks = [t for t, lst in unseen_by_task.items() if lst]
        if not available_tasks:
            print("Error: No more datasets available.")
            return None

        # step 1: pick a task uniformly
        task = random.choice(available_tasks)
        # step 2: pick an ID uniformly from that task’s unseen pool
        rnd_id = int(random.choice(unseen_by_task[task]))
        # now fetch it
        return cls.fetch_by_id(rnd_id)

    @classmethod
    def fetch_random_tasks(cls, task, k=1):
        """
        Fetch exactly k valid datasets for the specified task(s), choosing
        among task‐types uniformly and within each task uniformly.
        """
        # normalize tasks to list
        tasks = [task] if isinstance(task, str) else list(task)
        # validate
        for t in tasks:
            if t not in cls.task_ids:
                raise ValueError(f"Invalid task type: {t}. Must be 'regression' or 'classification'.")

        datasets = []
        attempts = 0
        # we'll cap attempts to avoid infinite loops
        max_attempts = sum(len(cls.task_ids[t]) for t in tasks) * 2

        while len(datasets) < k:
            # build unseen_by_task for the requested subset
            unseen_by_task = {
                t: list(set(cls.task_ids[t]) - set(cls.used_ids))
                for t in tasks
            }
            # flatten to see if any remain
            if not any(unseen_by_task.values()):
                raise RuntimeError(
                    f"Insufficient datasets available. Requested {k}, but only found {len(datasets)}. "
                    "Consider resetting the used_ids list or reducing k."
                )

            # pick a task uniformly among those that still have unseen IDs
            available_tasks = [t for t, lst in unseen_by_task.items() if lst]
            chosen_task = random.choice(available_tasks)

            # pick an ID uniformly from that task
            rnd_id = int(random.choice(unseen_by_task[chosen_task]))
            ds = cls.fetch_by_id(rnd_id)
            if ds:
                datasets.append(ds)

            attempts += 1
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Unable to fetch {k} datasets after {attempts} attempts. "
                    f"Only found {len(datasets)} valid datasets."
                )

        return datasets
    

    @classmethod
    def save_used_ids(cls):
        """Save the used IDs to a CSV file."""
        pd.DataFrame({"tid": cls.used_ids}).to_csv(cls.BASE_PATH + "used_ids.csv", index=False)

    @classmethod
    def reset_used_ids(cls):
        """Reset the list of used dataset IDs."""
        cls.used_ids = []
        if os.path.exists(cls.BASE_PATH + "used_ids.csv"):
            os.remove(cls.BASE_PATH + "used_ids.csv")