import os
import pathlib
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from wandb.sklearn import plot_feature_importances

import wandb
from misc import get_train_data

entity = "hka-ml1"
project = "DMML1_Projekt_Tim"

sweep_configuration = {
    "method": "grid",
    "name": "Predicting Sales with Store ID",
    "metric": {"goal": "maximize", "name": "mean_val_score"},
    "parameters": {
        "model": {"values": ["random_forest", "gradient_boosting"]},
        "n_estimators": {"values": [10, 50, 100, 200, 400]},
        "val_size": {"values": [0.2, 0.3]},
        "n_folds": {"values": [2, 5]},
        "max_depth": {"values": [3, 5, 10]},
        "max_features": {"values": [1, 2, 4, 8, "sqrt", "log2"]},
    },
}

# Make sure the model_output directory exists
pathlib.Path("./model_output").mkdir(parents=True, exist_ok=True)

is_leader = False
if pathlib.Path("./.sweep_id").exists():
    with open("./.sweep_id", "r") as f:
        sweep_id = f.read().strip()
else:
    sweep_id = wandb.sweep(sweep_configuration, project=project, entity=entity)
    with open("./.sweep_id", "w") as f:
        is_leader = True
        f.write(f"{entity}/{project}/{sweep_id}")

print(f"Leader: {is_leader}")
print(f"Sweep ID: {sweep_id}")


def main():
    features, target = get_train_data()

    run = wandb.init(tags=[target.name])

    print(f"Features: {features.columns}")
    print(f"Target: {target.name}")
    print(f"Number of features: {len(features.columns)}")
    print(f"Number of samples: {len(features)}")

    run.log_code("./", name=f"sweep-{run.sweep_id}-code", include_fn=lambda path: path.endswith(".py"))

    kfold = KFold(n_splits=run.config.n_folds, shuffle=True, random_state=42)

    features, x_val, target, y_val = train_test_split(features, target, test_size=run.config.val_size, random_state=42)

    test_scores = []
    val_scores = []
    train_scores = []
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

        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model.fit(x_train, y_train)
        val_score = model.score(x_val, y_val)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        test_scores.append(test_score)
        val_scores.append(val_score)
        train_scores.append(train_score)

        run.log({
            "split": index,
            "test_score": test_score,
            "val_score": val_score,
            "train_score": train_score,
        })
        plot_feature_importances(model, features.columns)
        with open(f"./model_output/model-{run.id}-{index}.pkl", "wb") as f:
            pickle.dump(model, f)

        art = wandb.Artifact(f"model-{run.id}", type="model")
        art.add_file(f"./model_output/model-{run.id}-{index}.pkl", name=f"model-{index}.pkl")
        run.log_artifact(art)

    run.log({
        "mean_test_score": sum(test_scores) / len(test_scores),
        "mean_val_score": sum(val_scores) / len(val_scores),
        "mean_train_score": sum(train_scores) / len(train_scores),
    })

    run.finish()


if __name__ == "__main__":
    wandb.agent(sweep_id, main)
