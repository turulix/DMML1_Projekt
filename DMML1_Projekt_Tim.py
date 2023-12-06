from sklearn.ensemble import RandomForestRegressor

from misc import get_train_data

customer_forrest = RandomForestRegressor()

rawdata, full_data, x_train, x_test, y_train, y_test = get_train_data(preCustomers=True)

customer_forrest.fit(x_train, y_train)
print(customer_forrest.score(x_test, y_test))

sales_forrest = RandomForestRegressor()
rawdata["PredictedCustomers"] = customer_forrest.predict(full_data)

rawdata, full_data, x_train, x_test, y_train, y_test = get_train_data(rawdata=rawdata, preCustomers=False)

sales_forrest.fit(x_train, y_train)
print(sales_forrest.score(x_test, y_test))
