import os
import pathlib
import pickle

import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from wandb.sklearn import plot_feature_importances


def process_data(data: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes the data. It joins the stores data with the train data.
    It also adds new features like Year and Month and fills NaN values with 0.
    """

    stores["CompetitionOpenSinceYear"].fillna(round(stores.CompetitionOpenSinceYear.mean()))
    stores["CompetitionOpenSinceMonth"].fillna(round(stores.CompetitionOpenSinceMonth.mean()))

    stores.fillna(0, inplace=True)

    data = data.merge(stores, on="Store ID")
    # data = data.loc[data["Open"] == 1]
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    return data


def get_train_data(use_train: bool = True) -> (pd.DataFrame, pd.Series):
    # Load stores & Trainings Data.
    stores = pd.read_csv("../data/dmml1_stores.csv")
    train_data = pd.read_csv("../data/dmml1_train.csv")

    # Process the data, join the stores data with the train data.
    # Add new Features, like Year and Month.
    # Also fill NaN values with 0.
    train_data = process_data(train_data, stores)

    # Split the data into features and target and drop columns that are meant to be predicted.
    features = train_data.drop(columns=["Sales", "Customers"])
    target = train_data["Sales"]

    # Create a ColumnTransformer to transform the data.
    column_transformer = ColumnTransformer([
        ("Drop Unused", "drop", [
            "Date",
            "PromoInterval"
        ]),
        ("One Hot Encode", OneHotEncoder(handle_unknown="ignore"), [
            "StateHoliday",
            "StoreType",
            "Assortment"
        ]),
        ("Scale", StandardScaler(), [
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "Year",
            "Month"
        ]),
    ], remainder="passthrough")

    # Fit the transformer on the feature dataframe.
    # Reason for this is that the transformer needs to stay consistent between training and testing.
    # We don't want to process the data differently between training and testing.
    column_transformer.fit(features)

    if use_train:
        # Transform the trainings data and drop rows with NaN values.
        transformed_train_data = pd.DataFrame(
            column_transformer.transform(features),
            columns=column_transformer.get_feature_names_out()
        )
        # This should do nothing, but just to be sure.
        transformed_train_data.dropna(inplace=True)  # Drop rows with NaN values
        return transformed_train_data, target
    else:
        test_data = pd.read_csv("../data/dmml1_test.csv")
        test_data = process_data(test_data, stores)
        transformed_test_data = pd.DataFrame(
            column_transformer.transform(test_data),
            columns=column_transformer.get_feature_names_out()
        )
        return transformed_test_data, None


def maybe_start_sweep(sweep_configuration, project, entity) -> str:
    """
    This function is to easily parallelize the sweeps.
    A sweep is a hyperparameter search, where multiple models are trained with different hyperparameters.
    Basically a grid search, but with more features in our case.
    """

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

    return sweep_id


def train_and_evaluate(model, features: pd.DataFrame, target: pd.Series, run):
    """This function trains and evaluates the model. Just to keep the main function clean."""

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

        # Plot the feature importance.
        # This is a helper function from wandb. And checks if the model has a feature_importance_ attribute.
        # If it does, it will plot the feature importance.
        plot_feature_importances(model, features.columns)

        # Save the model to a file.
        with open(f"./model_output/model-{run.id}-{index}.pkl", "wb") as f:
            pickle.dump(model, f)

            if os.stat(f"./model_output/model-{run.id}-{index}.pkl").st_size > 30 * 1024 * 1024:
                print("Model too big, skipping upload")
                continue

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
