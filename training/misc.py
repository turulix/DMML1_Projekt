import os
import pathlib
import pickle

import numpy as np
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

    stores["CompetitionOpenSinceYear"].fillna(round(stores["CompetitionOpenSinceYear"].mean()), inplace=True)
    stores["CompetitionOpenSinceMonth"].fillna(round(stores["CompetitionOpenSinceMonth"].mean()), inplace=True)

    data = data.merge(stores, on="Store ID")

    data["CompetitionDistance"].fillna(data["CompetitionDistance"].mean(), inplace=True)
    data["Promo2SinceWeek"].fillna(0, inplace=True)
    data["Promo2SinceYear"].fillna(0, inplace=True)
    data["PromoInterval"].fillna("", inplace=True)

    data['BeforeHoliday'] = np.where((data['Open'] == 1) & ((
            data['StateHoliday'].shift(1).isin(['a', 'b', 'c']) | data['StateHoliday'].shift(2).isin(
        ['a', 'b', 'c']))), 1, 0)

    data['AfterHoliday'] = np.where((data['Open'] == 1) & (
        (data['StateHoliday'].shift(-1).isin(['a', 'b', 'c']) | data['StateHoliday'].shift(-2).isin(['a', 'b', 'c']))),
                                    1, 0)

    data.fillna(0, inplace=True)

    # data = data.loc[data["Open"] == 1]
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month

    return data


def get_train_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # Load stores & Trainings Data.
    stores = pd.read_csv("../data/dmml1_stores.csv")

    # Process the data, join the stores data with the train data.
    # Add new Features, like Year and Month.
    # Also fill NaN values with 0.
    train_data = process_data(train_data, stores)
    test_data = process_data(test_data, stores)

    # Split the data into features and target and drop columns that are meant to be predicted.
    train_features = train_data.drop(columns=["Sales", "Customers"])
    train_target = train_data["Sales"]

    test_features = test_data.drop(columns=["Sales", "Customers"])
    test_target = test_data["Sales"]

    # Create a ColumnTransformer to transform the data.
    column_transformer = ColumnTransformer([
        ("Drop Unused", "drop", [
            "Date",
        ]),
        ("One Hot Encode", OneHotEncoder(handle_unknown="ignore"), [
            "StateHoliday",
            "StoreType",
            "Assortment",
            "PromoInterval"
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
    column_transformer.fit(train_features)

    return (
        pd.DataFrame(column_transformer.transform(train_features), columns=column_transformer.get_feature_names_out()),
        train_target,
        pd.DataFrame(column_transformer.transform(test_features), columns=column_transformer.get_feature_names_out()),
        test_target
    )

    # if use_train:
    #     # Transform the trainings data and drop rows with NaN values.
    #     transformed_train_data = pd.DataFrame(
    #         column_transformer.transform(train_features),
    #         columns=column_transformer.get_feature_names_out()
    #     )
    #     # This should do nothing, but just to be sure.
    #     transformed_train_data.dropna(inplace=True)  # Drop rows with NaN values
    #     return transformed_train_data, train_target
    # else:
    #     test_data = pd.read_csv("../data/dmml1_test.csv")
    #     test_data = process_data(test_data, stores)
    #     transformed_test_data = pd.DataFrame(
    #         column_transformer.transform(test_data),
    #         columns=column_transformer.get_feature_names_out()
    #     )
    #     return transformed_test_data, None


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


def train_and_evaluate(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, run):
    """This function trains and evaluates the model. Just to keep the main function clean."""

    print(f"Features: {x_train.columns}")
    print(f"Target: {y_train.name}")
    print(f"Number of features: {len(x_train.columns)}")
    print(f"Number of samples: {len(x_train)}")

    # Upload the current code to wandb.
    run.log_code("./", name=f"sweep-{run.sweep_id}-code", include_fn=lambda path: path.endswith(".py"))

    # Create a KFold cross validator.
    kfold = KFold(n_splits=run.config.n_folds, shuffle=True, random_state=42)

    # Split the data into trainings data & validation data.
    test_scores = []
    val_scores = []
    train_scores = []

    # Iterate over the splits.
    for index, (train_index, val_index) in enumerate(kfold.split(x_train, y_train)):
        # Get the trainings and validation data for this split.
        x_train_real, x_val = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_real, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Fit the model on the training's data.
        model.fit(x_train_real, y_train_real)

        # Evaluate the model on the different datasets.
        train_score = model.score(x_train_real, y_train_real)
        val_score = model.score(x_val, y_val)
        test_score = model.score(x_test, y_test)

        # Store the scores. To calculate the mean later.
        val_scores.append(val_score)
        train_scores.append(train_score)
        test_scores.append(test_score)

        # Log the scores and the model. To WandB.
        run.log({
            "split": index,
            "val_score": val_score,
            "train_score": train_score,
            "test_score": test_score,
        })

        # Plot the feature importance.
        # This is a helper function from wandb. And checks if the model has a feature_importance_ attribute.
        # If it does, it will plot the feature importance.
        plot_feature_importances(model, x_train_real.columns)

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
        "mean_val_score": sum(val_scores) / len(val_scores),
        "mean_train_score": sum(train_scores) / len(train_scores),
        "mean_test_score": sum(test_scores) / len(test_scores),
    })

    # Finish the run.
    run.finish()
