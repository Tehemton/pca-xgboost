from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
import os
import pprint
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
from math import sqrt
import matplotlib.pyplot as plt
import pickle


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

#! neer scale the dataset like done in the commented code below
#! scaling beforehand causes data leakage into the predictions
# * three different sclers are implemented to experiment
# ss = StandardScaler()
# df1 = pd.DataFrame(ss.fit_transform(df1), index=df1.index, columns=df1.columns)
# mms = MinMaxScaler()
# df1 = pd.DataFrame(mms.fit_transform(
#     df1), index=df1.index, columns=df1.columns)
# rbs = RobustScaler()
# df1 = pd.DataFrame(rbs.fit_transform(
#     df1), index=df1.index, columns=df1.columns)

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

#! converting target to numpy array and rehaping since scalerneeds it that way
Ytrain = Ytrain.to_numpy()
Ytrain = Ytrain.reshape((-1, 1))

# * scale train X, Y and test X
rbsxt = RobustScaler()
rbsyt = RobustScaler()
rbsxte = RobustScaler()

Xtrain = pd.DataFrame(rbsxt.fit_transform(
    Xtrain), index=Xtrain.index, columns=Xtrain.columns)
Ytrain = pd.DataFrame(rbsyt.fit_transform(
    Ytrain))  # , index=Ytrain.index, columns=Ytrain.columns)
Xtest = pd.DataFrame(rbsxte.fit_transform(
    Xtest), index=Xtest.index, columns=Xtest.columns)

#! do PCA here to reduce dimensionality
pca = decomposition.PCA(n_components=10)
pca.fit(Xtrain)
scoresXtr = pd.DataFrame(pca.transform(Xtrain), columns=[i for i in range(10)])
explained_variance = pca.explained_variance_ratio_
explained_variance = np.insert(explained_variance, 0, 0)
cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))
pca.fit(Xtest)
scoresXte = pd.DataFrame(pca.transform(Xtest), columns=[i for i in range(10)])
# * after printing we can see that
# * almost the entire data (98.6% variance)
# * can be reduced to 6 dimensions
print(cumulative_variance)

#! unomment for tuning hyperparameters
# model = XGBRegressor(n_estimators=1000,
#                      reg_lambda=1,
#                      gamma=0,
#                      max_depth=3)

if not os.path.isfile('.\\model.dat'):
    model = XGBRegressor()
    model.fit(scoresXtr, Ytrain)
    pickle.dump(model, open("model.dat", "wb"))

else:
    model = pickle.load(open("model.dat", "rb"))
ypred = model.predict(scoresXte)
ypred = ypred.reshape((-1, 1))

ypred = rbsyt.inverse_transform(ypred)
rmse = sqrt(mean_squared_error(Ytest, ypred))
print(rmse)
#!the model was trained by taking 6, 8 and 10 components
#!the best RMSE of 12.31 was observed with 10 components
#! * components gave an RMSE of 12.43

print(model.get_booster().get_fscore())


# fig = px.line(Ytest,  title='Ytest')
Ytest_np = Ytest.to_numpy()
ypred = ypred.flatten()

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
