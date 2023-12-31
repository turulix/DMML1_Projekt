import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from misc import get_train_data, maybe_start_sweep, train_and_evaluate

model_type = "Ensemble"
entity = "hka-ml1"
project = "DMML1_Projekt_Tim"

# Settings for the WandB Sweep, we are using a grid search here.
sweep_configuration = {
    "method": "grid",
    "name": f"{model_type} - Predicting Sales with PromoInterval & learning_rate & Better",
    "metric": {"goal": "maximize", "name": "mean_test_score"},
    "parameters": {
        #  "model": {"values": ["random_forest", "gradient_boosting"]},
        "n_estimators": {"values": [100, 200, 400]},
        "max_depth": {"values": [3, 5, 10]},
        "max_features": {"values": [16, 20, 28]},
        "learning_rate": {"values": [0.2, 0.3, 0.1]},

        # The following parameters are not used by the model, but are used by the training script.
        "val_size": {"values": [0.2]},
        "n_folds": {"values": [5]},
    },
}

sweep_id = maybe_start_sweep(sweep_configuration, project, entity)


def main():
    train_data = pd.read_csv("../data/dmml1_train.csv")
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    x_train, y_train, x_test, y_test = get_train_data(train_data, test_data)
    # Initialize wandb.
    run = wandb.init()

    # if run.config.model == "random_forest":
    #     # Create a RandomForestRegressor with the parameters from the sweep.
    #     model = RandomForestRegressor(
    #         n_estimators=run.config.n_estimators,
    #         max_depth=run.config.max_depth,
    #         max_features=run.config.max_features,
    #     )
    # elif run.config.model == "gradient_boosting":
    #     # Create a GradientBoostingRegressor with the parameters from the sweep.
    #     model = GradientBoostingRegressor(
    #         n_estimators=run.config.n_estimators,
    #         max_depth=run.config.max_depth,
    #         max_features=run.config.max_features,
    #     )
    # else:
    #     raise ValueError("Invalid model")

    model = GradientBoostingRegressor(
        n_estimators=run.config.n_estimators,
        learning_rate=run.config.learning_rate,
        max_depth=run.config.max_depth,
        max_features=run.config.max_features,
    )

    train_and_evaluate(model, x_train, y_train, x_test, y_test, run)


# Start the WandB sweep agent.
if __name__ == "__main__":
    wandb.agent(sweep_id, main)
