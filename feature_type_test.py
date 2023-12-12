import numpy as np
import pandas as pd

df_stores = pd.read_csv('./data/dmml1_stores.csv')
df_test = pd.read_csv('./data/dmml1_test.csv')
df_train = pd.read_csv('./data/dmml1_train.csv')


df_train['AfterHoliday'] = np.where((df_train['Open'] == 1) & ((df_train['StateHoliday'].shift(-1).isin(['a', 'b', 'c']) | df_train['StateHoliday'].shift(-2).isin(['a', 'b', 'c']))), 1, 0)

df_train['BeforeHoliday'] = np.where((df_train['Open'] == 1) & ((df_train['StateHoliday'].shift(1).isin(['a', 'b', 'c']) | df_train['StateHoliday'].shift(2).isin(['a', 'b', 'c']))), 1, 0)

print(df_train)