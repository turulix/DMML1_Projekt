import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_train_data(rawdata=None) -> (pd.DataFrame, pd.Series):
    """
    :param rawdata: The data to use. If None, the data will be loaded from the csv files.
    :return: The data, The Processed Data, the train and test data for the x values and the train and test data for the y values.
    """
    if rawdata is None:
        stores = pd.read_csv("../data/dmml1_stores.csv")
        rawdata = pd.read_csv("../data/dmml1_train.csv")
        rawdata = rawdata.merge(stores, on="Store ID")
        rawdata = rawdata.loc[rawdata["Open"] == 1]
        rawdata["Date"] = pd.to_datetime(rawdata["Date"])

    column_transformer = ColumnTransformer([
        ("Drop Unused", "drop", [
            # "Store ID",
            "Date",
            "Open",
            "PromoInterval"
        ]),
        ("One Hot Encode", OneHotEncoder(handle_unknown="ignore"), ["StateHoliday", "StoreType", "Assortment"]),
        ("Scale", StandardScaler(), [
            # "Customers",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2SinceWeek",
            "Promo2SinceYear",
        ]),
    ], remainder="passthrough")
    rawdata.dropna(inplace=True)
    data = pd.DataFrame(column_transformer.fit_transform(rawdata), columns=column_transformer.get_feature_names_out())

    return (
        # data.drop(columns=["remainder__Customers", "remainder__Sales"]),
        data.drop(columns=["remainder__Sales", "remainder__Customers"]),
        data["remainder__Sales"]
    )
