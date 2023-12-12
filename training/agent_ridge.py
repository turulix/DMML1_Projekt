import pandas as pd
import wandb
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from misc import get_train_data, maybe_start_sweep, train_and_evaluate

model_type = "Ridge"
entity = "hka-ml1"
project = "DMML1_Projekt_Tim"

# Settings for the WandB Sweep
sweep_configuration = {
    "method": "grid",
    "name": f"{model_type} - Predicting Sales with Date & More Features",
    "metric": {"goal": "maximize", "name": "mean_val_score"},
    "parameters": {
        "alpha": {"values": [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]},
        "max_iter": {"values": [500, 1000, 2000, 4000, 8000, 16000]},

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

    model = Ridge(
        alpha=run.config.alpha,
        max_iter=run.config.max_iter,
    )

    train_and_evaluate(model, x_train, y_train, x_test, y_test, run)


# Start the WandB sweep agent.
if __name__ == "__main__":
    wandb.agent(sweep_id, main)
