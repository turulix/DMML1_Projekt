import pickle

from sklearn.ensemble import GradientBoostingRegressor

from training.misc import get_train_data

with open("../data/model-1.pkl", "rb") as f:
    model: GradientBoostingRegressor = pickle.load(f)

features, target = get_train_data(use_train=True)
features2, target2 = get_train_data(use_train=False)

print(model.predict(features2))
