import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
            # "Store ID",
            "Date",
            # "Open",
            "PromoInterval"
        ]),
        ("One Hot Encode", OneHotEncoder(handle_unknown="ignore"), [
            "StateHoliday",
            "StoreType",
            "Assortment"
        ]),
        ("Scale", StandardScaler(), [
            # "Customers",
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


def process_data(data: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:

    stores["CompetitionOpenSinceYear"].fillna(round(stores.CompetitionOpenSinceYear.mean()))
    stores["CompetitionOpenSinceMonth"].fillna(round(stores.CompetitionOpenSinceMonth.mean()))

    stores.fillna(0, inplace=True)

    data = data.merge(stores, on="Store ID")
    # data = data.loc[data["Open"] == 1]
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month


    return data
