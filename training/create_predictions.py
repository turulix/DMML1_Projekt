import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from training.misc import get_train_data

with open("../data/model_ensemble.pkl", "rb") as f:
    model: GradientBoostingRegressor = pickle.load(f)

features, target = get_train_data(use_train=True)

features, x_val, target, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

predictions = model.predict(x_val)
print(mean_absolute_error(y_val, predictions))
print(mean_squared_error(y_val, predictions))
print(mean_absolute_percentage_error(y_val, predictions))
print(explained_variance_score(y_val, predictions))
print(median_absolute_error(y_val, predictions))
print(r2_score(y_val, predictions))
print(max_error(y_val, predictions))

x_val.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

with_predictions = pd.concat([x_val, y_val, pd.DataFrame(predictions, columns=["Prediction"])], axis=1)
with_predictions["Error"] = with_predictions["Prediction"] - with_predictions["Sales"]

# Square the error to get rid of negative values.
with_predictions["Error"] = with_predictions["Error"] ** 2
# Take the square root to get the RMSE.
with_predictions["Error"] = with_predictions["Error"] ** 0.5

with_predictions["Error %"] = with_predictions["Error"] / with_predictions["Sales"] * 100

# Filter out the rows where Sales is 0.
with_predictions = with_predictions.loc[with_predictions["Sales"] != 0]

print(with_predictions.head(10))
