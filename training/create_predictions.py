import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from training.misc import get_train_data


def main():
    train_data = pd.read_csv("../data/dmml1_train.csv")
    test_data = pd.read_csv("../data/dmml1_test.csv")

    # Set Sales and Customers to 0 because they are not in the test data, and are dropped in get_train_data
    test_data["Sales"] = 0
    test_data["Customers"] = 0

    x_train, y_train, x_test, y_test = get_train_data(train_data, test_data)

    model: GradientBoostingRegressor = pickle.load(open("../data/model_ensemble.pkl", "rb"))

    predictions = model.predict(x_test)

    print(predictions)

    test_data["Sales"] = predictions

    # Set Sales to 0 where Open is 0
    test_data["Sales"] = np.where(test_data["Open"] == 0, 0, test_data["Sales"])

    test_data.drop(columns=["Customers"], inplace=True)

    test_data.to_csv("../data/dmml1_test_predictions.csv", index=False)




if __name__ == "__main__":
    main()
