import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def check_dataset_compatibility(dataset_id):
    """Check if a dataset is compatible with our requirements."""
    try:
        # Download the dataset
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )
        
        # Check 1: Ensure X is 2D and y is 1D
        if len(X.shape) != 2 or len(y.shape) != 1:
            return False
        
        # Check 2: Minimum number of samples to avoid n_splits issues
        if len(X) < 50:
            return False
        
        # Check 3: Features should not be too close to sample count (avoid overfitting)
        if X.shape[1] / X.shape[0] > 0.5:
            return False
            
        # Check 4: For regression tasks, ensure y is continuous (not categorical)
        unique_y = np.unique(y)
        if len(unique_y) < 10 and len(unique_y) / len(y) < 0.05:
            # If few unique values relative to dataset size, likely categorical
            return False
        
        # Check 5: Avoid sparse matrices that cause dtype errors
        if hasattr(X, 'toarray') or hasattr(y, 'toarray'):
            return False
            
        # Check 6: Test if train_test_split works (catches various edge cases)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return True
        except Exception:
            return False
            
    except Exception as e:
        print(f"Error checking dataset {dataset_id}: {str(e)}")
        return False


# Main code
# Download a list of tasks from OpenML
all_tasks = openml.tasks.list_tasks(output_format='dataframe')

# Only keep tasks that have associated datasets
all_tasks = all_tasks[~all_tasks['did'].isna()]

# Apply basic filters first
filtered = all_tasks[
    (all_tasks['NumberOfInstances'] >= 50) &   # Minimum 50 samples
    (all_tasks['NumberOfInstances'] <= 50000) & # Maximum 50000 samples
    (all_tasks['NumberOfFeatures'] <= 1000) &   # Maximum 1000 features
    # New condition: Ensure feature count isn't too close to sample count
    (all_tasks['NumberOfFeatures'] / all_tasks['NumberOfInstances'] < 0.5)
]

# Filter by task type
regression_tasks = filtered[filtered['task_type'] == 'Supervised Regression']
classification_tasks = filtered[filtered['task_type'] == 'Supervised Classification']

# Get the task and dataset IDs
regression_df = regression_tasks.reset_index()[['tid', 'did']]
classification_df = classification_tasks.reset_index()[['tid', 'did']]

# Further filter by checking actual dataset compatibility
print("Checking regression datasets compatibility...")
compatible_reg_dids = []
for did in regression_df['did'].unique()[:100]:  # Limit initial check to first 100
    if check_dataset_compatibility(int(did)):
        compatible_reg_dids.append(did)
        
print("Checking classification datasets compatibility...")
compatible_cls_dids = []
for did in classification_df['did'].unique()[:100]:  # Limit initial check to first 100
    if check_dataset_compatibility(int(did)):
        compatible_cls_dids.append(did)

# Filter dataframes to only include compatible datasets
regression_df = regression_df[regression_df['did'].isin(compatible_reg_dids)]
classification_df = classification_df[classification_df['did'].isin(compatible_cls_dids)]

# Save to CSV with 'tid' and 'did' columns
regression_df.to_csv('reg_ids.csv', index=False)
classification_df.to_csv('cls_ids.csv', index=False)

print(f"Saved {len(regression_df)} compatible regression tasks")
print(f"Saved {len(classification_df)} compatible classification tasks")