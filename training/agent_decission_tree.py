import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from misc import get_train_data, maybe_start_sweep, train_and_evaluate

model_type = "Decision Tree"
entity = "hka-ml1"
project = "DMML1_Projekt_Tim"

# Settings for the WandB Sweep, we are using a grid search here.
sweep_configuration = {
    "method": "grid",
    "name": f"{model_type} - Predicting Sales & Mean Competition & Open Included",
    "metric": {"goal": "maximize", "name": "mean_val_score"},
    "parameters": {
        "max_depth": {"values": [None, 3, 5, 10, 15, 20]},
        "max_features": {"values": [1, 2, 4, 8, 12, 16, None, "sqrt", "log2"]},
        "min_samples_split": {"values": [2, 5, 10]},
        "min_samples_leaf": {"values": [1, 2, 4, 8, 12, 16]},

        # The following parameters are not used by the model, but are used by the training script.
        "val_size": {"values": [0.2, 0.3]},
        "n_folds": {"values": [2, 5]},
    },
}

sweep_id = maybe_start_sweep(sweep_configuration, project, entity)


def main():
    train_data = pd.read_csv("../data/dmml1_train.csv")
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    x_train, y_train, x_test, y_test = get_train_data(train_data, test_data)
    # Initialize wandb.
    run = wandb.init()

    model = DecisionTreeRegressor(
        max_depth=run.config.max_depth,
        max_features=run.config.max_features,
        min_samples_split=run.config.min_samples_split,
        min_samples_leaf=run.config.min_samples_leaf,
    )

    train_and_evaluate(model, x_train, y_train, x_test, y_test, run)


# Start the WandB sweep agent.
if __name__ == "__main__":
    wandb.agent(sweep_id, main)
