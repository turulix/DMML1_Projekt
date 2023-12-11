# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split, KFold
#
# from training.misc import get_train_data
# import pickle
#
# if __name__ == "__main__":
#     # Get the train data.
#     features, target = get_train_data(use_train=True)
#
#     # Split the data into trainings data & validation data.
#     features, x_val, target, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
#
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#
#     val_scores = []
#     train_scores = []
#     test_scores = []
#
#     # Iterate over the splits.
#     for index, (train_index, test_index) in enumerate(kfold.split(features, target)):
#         model = LinearRegression()
#
#         # Get the trainings and test data for this split.
#         x_train, x_test = features.iloc[train_index], features.iloc[test_index]
#         y_train, y_test = target.iloc[train_index], target.iloc[test_index]
#
#         # Fit the model on the training's data.
#         model.fit(x_train, y_train)
#
#         # Evaluate the model on the different datasets.
#         val_score = model.score(x_val, y_val)
#         train_score = model.score(x_train, y_train)
#         test_score = model.score(x_test, y_test)
#
#         val_scores.append(val_score)
#         train_scores.append(train_score)
#         test_scores.append(test_score)
#
#         print(f"Split {index}")
#         print(f"Train: {len(train_index)}")
#         print(f"Test: {len(test_index)}")
#         print(f"Test score: {test_score}")
#         print(f"Val score: {val_score}")
#         print(f"Train score: {train_score}")
#
#     print("-----------------------------------")
#     print(f"Val: {sum(val_scores) / len(val_scores)}")
#     print(f"Train: {sum(train_scores) / len(train_scores)}")
#     print(f"Test: {sum(test_scores) / len(test_scores)}")
#
#     with open("../data/model_linear_regression.pkl", "wb") as f:
#         pickle.dump(model, f)
