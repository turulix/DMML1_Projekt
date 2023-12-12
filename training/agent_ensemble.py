import wandb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from misc import get_train_data, maybe_start_sweep, train_and_evaluate

model_type = "Ensemble"
entity = "hka-ml1"
project = "DMML1_Projekt_Tim"

# Settings for the WandB Sweep, we are using a grid search here.
sweep_configuration = {
    "method": "grid",
    "name": f"{model_type} - Predicting Sales with PromoInterval & learning_rate",
    "metric": {"goal": "maximize", "name": "mean_val_score"},
    "parameters": {
        #  "model": {"values": ["random_forest", "gradient_boosting"]},
        "n_estimators": {"values": [100, 200, 400]},
        "max_depth": {"values": [3, 5, 10]},
        "max_features": {"values": [16, 20, 28]},
        "learning_rate": {"values": [0.1, 0.01, 0.001, 0.0001]},

        # The following parameters are not used by the model, but are used by the training script.
        "val_size": {"values": [0.2]},
        "n_folds": {"values": [5]},
    },
}

sweep_id = maybe_start_sweep(sweep_configuration, project, entity)


def main():
    features, target = get_train_data(use_train=True)
    # Initialize wandb.
    run = wandb.init(tags=[target.name])

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

    train_and_evaluate(model, features, target, run)


# Start the WandB sweep agent.
if __name__ == "__main__":
    wandb.agent(sweep_id, main)
