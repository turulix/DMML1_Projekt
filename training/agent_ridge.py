import wandb
from sklearn.linear_model import Ridge

from misc import get_train_data
from agent import maybe_start_sweep, train_and_evaluate

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
    features, target = get_train_data(use_train=True)
    # Initialize wandb.
    run = wandb.init(tags=[target.name])

    model = Ridge(
        alpha=run.config.alpha,
        max_iter=run.config.max_iter,
    )

    train_and_evaluate(model, features, target, run)


# Start the WandB sweep agent.
if __name__ == "__main__":
    wandb.agent(sweep_id, main)
