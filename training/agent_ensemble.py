import os
import pathlib
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from wandb.sklearn import plot_feature_importances

import wandb
from misc import get_train_data

model_type = "Ensemble"
entity = "hka-ml1"
project = "DMML1_Projekt_Tim"

# Settings for the WandB Sweep
sweep_configuration = {
    "method": "grid",
    "name": f"{model_type} - Predicting Sales with Date & More Features",
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

# Make sure the model_output directory exists, as we will save the models there.
pathlib.Path("./model_output").mkdir(parents=True, exist_ok=True)

# Check if we are the leader of the sweep (The Agent that created it).
is_leader = False
if pathlib.Path("./.sweep_id").exists():
    with open("./.sweep_id", "r") as f:
        sweep_id = f.read().strip()
else:
    # We are the leader, so create the sweep.
    sweep_id = wandb.sweep(sweep_configuration, project=project, entity=entity)
    # Save the sweep id to a file, so we can check if we are the leader next time.
    with open("./.sweep_id", "w") as f:
        is_leader = True
        f.write(f"{entity}/{project}/{sweep_id}")

print(f"Leader: {is_leader}")
print(f"Sweep ID: {sweep_id}")


def main():
    features, target = get_train_data(use_train=True)
    # Initialize wandb.
    run = wandb.init(tags=[target.name])

    print(f"Features: {features.columns}")
    print(f"Target: {target.name}")
    print(f"Number of features: {len(features.columns)}")
    print(f"Number of samples: {len(features)}")

    # Upload the current code to wandb.
    run.log_code("./", name=f"sweep-{run.sweep_id}-code", include_fn=lambda path: path.endswith(".py"))

    # Create a KFold cross validator.
    kfold = KFold(n_splits=run.config.n_folds, shuffle=True, random_state=42)

    # Split the data into trainings data & validation data.
    features, x_val, target, y_val = train_test_split(features, target, test_size=run.config.val_size, random_state=42)

    test_scores = []
    val_scores = []
    train_scores = []

    # Iterate over the splits.
    for index, (train_index, test_index) in enumerate(kfold.split(features, target)):
        print(f"Split {index}")
        print(f"Train: {len(train_index)}")
        print(f"Test: {len(test_index)}")
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

        # Get the trainings and test data for this split.
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Fit the model on the training's data.
        model.fit(x_train, y_train)

        # Evaluate the model on the different datasets.
        val_score = model.score(x_val, y_val)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        # Store the scores. To calculate the mean later.
        test_scores.append(test_score)
        val_scores.append(val_score)
        train_scores.append(train_score)

        # Log the scores and the model. To WandB.
        run.log({
            "split": index,
            "test_score": test_score,
            "val_score": val_score,
            "train_score": train_score,
        })

        # Plot the feature importances.
        # This is a helper function from wandb. And checks if the model has a feature_importances_ attribute.
        # If it does, it will plot the feature importances.
        plot_feature_importances(model, features.columns)

        # Save the model to a file.
        with open(f"./model_output/model-{run.id}-{index}.pkl", "wb") as f:
            pickle.dump(model, f)

        # Create an artifact for the model and upload it to wandb.
        art = wandb.Artifact(f"model-{run.id}", type="model")
        art.add_file(f"./model_output/model-{run.id}-{index}.pkl", name=f"model-{index}.pkl")
        run.log_artifact(art)

    # Log the mean scores.
    run.log({
        "mean_test_score": sum(test_scores) / len(test_scores),
        "mean_val_score": sum(val_scores) / len(val_scores),
        "mean_train_score": sum(train_scores) / len(train_scores),
    })

    # Finish the run.
    run.finish()


# Start the WandB sweep agent.
if __name__ == "__main__":
    wandb.agent(sweep_id, main)
