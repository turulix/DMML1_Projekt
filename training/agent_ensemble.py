import wandb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from misc import get_train_data
from agent import maybe_start_sweep, train_and_evaluate

model_type = "Ensemble"
entity = "hka-ml1"
project = "DMML1_Projekt_Tim"

# Settings for the WandB Sweep
sweep_configuration = {
    "method": "grid",
    "name": f"{model_type} - Predicting Sales & Mean Competition & Open Included",
    "metric": {"goal": "maximize", "name": "mean_val_score"},
    "parameters": {
        "model": {"values": ["random_forest", "gradient_boosting"]},
        "n_estimators": {"values": [10, 50, 100, 200, 400]},
        "val_size": {"values": [0.2, 0.3]},
        "n_folds": {"values": [2, 5]},
        "max_depth": {"values": [3, 5, 10]},
        "max_features": {"values": [1, 2, 4, 8, 12, 16, None, "sqrt", "log2"]},
    },
}

sweep_id = maybe_start_sweep(sweep_configuration, project, entity)


def main():
    features, target = get_train_data(use_train=True)
    # Initialize wandb.
    run = wandb.init(tags=[target.name])

    if run.config.model == "random_forest":
        model = RandomForestRegressor(
            n_estimators=run.config.n_estimators,
            max_depth=run.config.max_depth,
            max_features=run.config.max_features,

        )
    elif run.config.model == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=run.config.n_estimators,
            max_depth=run.config.max_depth,
            max_features=run.config.max_features,
        )
    else:
        raise ValueError("Invalid model")

    train_and_evaluate(model, features, target, run)


# Start the WandB sweep agent.
if __name__ == "__main__":
    wandb.agent(sweep_id, main)
