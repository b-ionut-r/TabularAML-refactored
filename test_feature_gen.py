import pandas as pd
import numpy as np
np.NaN = np.nan
from sklearn.datasets import make_classification
from tabularaml.generate.features import FeatureGenerator

def test_feature_generator():
    X, y = make_classification(n_samples=2000, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"num_{i}" for i in range(5)])
    
    # Add categorical and time columns
    X["cat_0"] = np.random.choice(["A", "B", "C"], size=2000)
    X["id_col"] = np.random.choice([1, 2, 3, 4, 5], size=2000)
    X["time_col"] = np.repeat(np.arange(400), 5)
    
    y = pd.Series(y)
    
    fg = FeatureGenerator(
        n_generations=2,
        n_children=10,
        cv=3,
        use_proxy_evaluation=True,
        proxy_top_pct=0.5,
        rotate_cv_folds=True,
        meta_validation_frac=0.1,
        final_selection=True,
        time_col="time_col",
        id_col="id_col"
    )
    
    X_new, pipeline, gen, interactions = fg.search(X, y)
    print(X_new.columns)
    print("Test finished successfully!")

if __name__ == "__main__":
    test_feature_generator()
