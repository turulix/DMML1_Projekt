import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from wandb.sklearn import plot_feature_importances

import wandb
from misc import get_train_data

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "mean_score"},
    "parameters": {
        "model": {"values": ["random_forest", "gradient_boosting"]},
        "n_estimators": {"values": [10, 50, 100, 200, 500]},
        "max_depth": {"values": [3, 5, 10, 20]},
        "max_features": {"values": [1, 2, 4, 8]},
    },
}

sweep_id = "hka-ml1/DMML1_Projekt_Tim/pu7bb2r6"
#sweep_id = wandb.sweep(sweep_configuration, project="DMML1_Projekt_Tim", entity="hka-ml1")

print(sweep_id)


def main():
    run = wandb.init()

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    features, target = get_train_data(preCustomers=True)
    scores = []
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
        score = model.score(x_test, y_test)
        scores.append(score)

        run.log({
            "split": index,
            "score": score,
        })
        plot_feature_importances(model, features.columns)
        with open(f"./model_output/model-{run.id}-{index}.pkl", "wb") as f:
            pickle.dump(model, f)

        run.log_model(f"./model_output/model-{run.id}-{index}.pkl", name=f"model-{run.id}-{index}.pkl")
    run.log({"mean_score": sum(scores) / len(scores)})

    run.finish()


if __name__ == "__main__":
    wandb.agent(sweep_id, main)
