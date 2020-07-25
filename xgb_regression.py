from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pprint
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
from math import sqrt
import matplotlib.pyplot as plt

#! this program will develop an XGBoost reqression model
#! we will then do feature importance and selecction
#! retrain the model on only the selected important features
#! compare RMSE for both results

# * load the data
li = []
for file in os.listdir('.\\emission_data'):
    imp = pd.read_csv(f'.\\emission_data\\{file}', index_col=None, header=0)
    li.append(imp)

df1 = pd.concat(li, axis=0, ignore_index=True)
# print(df1.head)

# * scale dataset 3 different scalers are implemented here
# ss = StandardScaler()
# df1 = pd.DataFrame(ss.fit_transform(df1), index=df1.index, columns=df1.columns)
# mms = MinMaxScaler()
# df1 = pd.DataFrame(mms.fit_transform(
#     df1), index=df1.index, columns=df1.columns)
rbs = RobustScaler()
df1 = pd.DataFrame(rbs.fit_transform(
    df1), index=df1.index, columns=df1.columns)

df1.plot(subplots=True, layout=(6, 2))
plt.show()
# print(df1.head)

# * split into X and Y
X = df1.iloc[:, 0:10]
Y = df1.iloc[:, 10]
# print(X.shape, Y.shape)

# * split sets
split = 0.6

Xtrain, Xtest = X.iloc[:int(len(X.index)*split),
                       :], X.iloc[int(len(X.index)*split):, :]
# print(Xtrain.shape, '\t', Xtest.shape)
Ytrain, Ytest = Y.iloc[:int(len(Y.index)*split)
                       ], Y.iloc[int(len(Y.index)*split):]
# print(Ytrain.shape, '\t', Ytest.shape)

#! unomment for tuning hyperparameters
# model = XGBRegressor(n_estimators=1000,
#                      reg_lambda=1,
#                      gamma=0,
#                      max_depth=3)

model = XGBRegressor()

model.fit(Xtrain, Ytrain)

ypred = model.predict(Xtest)
rmse = sqrt(mean_squared_error(Ytest, ypred))
print(rmse)


# fig = px.line(Ytest,  title='Ytest')
Ytest_np = Ytest.to_numpy()
ypred = np.array(ypred)
fig = go.Figure()
fig.add_trace(go.Scatter(y=Ytest_np,
                         mode='lines',
                         name='Ytest'))
fig.add_trace(go.Scatter(y=ypred,
                         mode='lines',
                         name='ypred'))
fig.write_html(f'.\\Ytest.html')

new = 2
chromepath = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
webbrowser.get(chromepath).open(f'.\\Ytest.html', new=new)
