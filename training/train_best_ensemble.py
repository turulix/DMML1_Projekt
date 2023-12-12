import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from training.misc import get_train_data

def main():
    features, target = get_train_data(use_train=True)

    params = {
        "val_size": 0.2,
        "n_estimators": 200,
        "max_depth": 10,
        "max_features": 16,
    }

    features, x_val, target, y_val = train_test_split(features, target, test_size=params["val_size"], random_state=42)

    gradient_model = GradientBoostingRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        max_features=params["max_features"],
    )

    gradient_model.fit(features, target)

    print(gradient_model.score(x_val, y_val))

    x_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    predictions = gradient_model.predict(x_val)

    with_predictions = pd.concat([x_val, y_val, pd.DataFrame(predictions, columns=["Prediction"])], axis=1)
    with_predictions["Error"] = with_predictions["Prediction"] - with_predictions["Sales"]
    # Set Penalty to 150 when Error > 3000
    # with_predictions["Penalty"] = with_predictions["Error"].loc[with_predictions["Error"] < -4000, "Error"] = 100
    # with_predictions["Penalty"] = with_predictions.loc[with_predictions["Error"] > 3000, "Error"] = 150
    # with_predictions["Penalty"] = with_predictions.loc[with_predictions["Error"] > 6000, "Error"] = 250
    # with_predictions["Penalty"] = with_predictions["Penalty"].fillna(0)

    with_predictions["Penalty"] = np.where(with_predictions["Error"] < -4000, 100, 0)
    with_predictions["Penalty"] = np.where(with_predictions["Error"] > 3000, 150, with_predictions["Penalty"])
    with_predictions["Penalty"] = np.where(with_predictions["Error"] > 6000, 250, with_predictions["Penalty"])

    with_predictions = with_predictions[with_predictions["remainder__Open"] == 1]

    print(with_predictions.head(10))

if __name__ == "__main__":
    main()
