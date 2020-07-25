from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df1 = pd.read_csv('crx.data', delimiter=',')
df1.columns = [n for n in range(1, 17)]
# print(df1.head())
# print(df1.dtypes)

# * instantiate label encoder and standard scaler
le = LabelEncoder()
ss = StandardScaler()
# * get colun names that have objects and mask them to a list
mask = df1.dtypes == object
categorical_cols = df1.columns[mask].tolist()

# * apply label encoder to the columns masked earlier
df1[categorical_cols] = df1[categorical_cols].apply(
    lambda col: le.fit_transform(col))

# * scale values using standard scaler
df1 = pd.DataFrame(ss.fit_transform(df1), index=df1.index, columns=df1.columns)

X = df1.iloc[:, 0:15]
Y = df1.iloc[:, 15]

seed = 2
test_split = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_split, random_state=seed)

Y_train, Y_test = [round(y) for y in Y_train], [round(y) for y in Y_test]
model = XGBClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

predictions = [round(y) for y in y_pred]
accuracy = accuracy_score(Y_test, predictions)

print(f'accuracy = {accuracy*100}%')
