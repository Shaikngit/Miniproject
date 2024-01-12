import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipeline
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.training_config.features],  # predictors
        data[config.training_config.target],
        test_size=config.training_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.training_config.random_state,
    )

    # Pipeline fitting
    bikeshare_pipeline.fit(X_train,y_train)
    y_pred = bikeshare_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)

    r2 = r2_score(y_test, y_pred)
    print("R-squared (R2):", r2)

    # persist trained model
    save_pipeline(pipeline_to_persist= bikeshare_pipeline)
    # printing the score
    
if __name__ == "__main__":
    run_training()