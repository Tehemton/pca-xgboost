from sklearn import datasets
from sklearn.preprocessing import scale, StandardScaler  # Data scaling
from sklearn import decomposition  # PCA
import pandas as pd  # pandass
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import webbrowser

iris = datasets.load_iris()
print(iris.feature_names)
print(iris.target_names)
X = iris.data
Y = iris.target
print(X.shape, Y.shape)

# here we cale the data standardscaler can ube used as well
X = scale(X)

# Here we define the number of PC to use as 3s
# * we first check with the number of dimensions and then fine tune the number of components we need
#! detailed below when we calculate explained variance
pca = decomposition.PCA(n_components=3)
pca.fit(X)

# Compute and retrieve the scores value
scores = pca.transform(X)
scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])
print(scores_df)


Y_label = []

for i in Y:
    if i == 0:
        Y_label.append('Setosa')
    elif i == 1:
        Y_label.append('Versicolor')
    else:
        Y_label.append('Virginica')

Species = pd.DataFrame(Y_label, columns=['Species'])
df_scores = pd.concat([scores_df, Species], axis=1)

# Retrieve the loadings values
loadings = pca.components_.T
df_loadings = pd.DataFrame(
    loadings, columns=['PC1', 'PC2', 'PC3'], index=iris.feature_names)
print(df_loadings)

#! Explained variance for each PC
# * for PCA, start with the number of dimension as the omponents and compute the explained variance and cumilative variances
# * inspect the cumilative variance dataframe and pick the number of components where >95% variance is captured
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

explained_variance = np.insert(explained_variance, 0, 0)
cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

pc_df = pd.DataFrame(['', 'PC1', 'PC2', 'PC3'], columns=['PC'])
explained_variance_df = pd.DataFrame(
    explained_variance, columns=['Explained Variance'])
cumulative_variance_df = pd.DataFrame(
    cumulative_variance, columns=['Cumulative Variance'])
df_explained_variance = pd.concat(
    [pc_df, explained_variance_df, cumulative_variance_df], axis=1)
print(df_explained_variance)

# ! plotting
counter = 1
new = 2
chromepath = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
fig = px.bar(df_explained_variance,
             x='PC', y='Explained Variance',
             text='Explained Variance',
             width=800)

fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.write_html(f'.\\test{counter}.html')
webbrowser.get(chromepath).open(f'.\\test{counter}.html', new=new)
counter = counter+1


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Cumulative Variance'],
        marker=dict(size=15, color="LightSeaGreen")
    ))

fig.add_trace(
    go.Bar(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Explained Variance'],
        marker=dict(color="RoyalBlue")
    ))

fig.write_html(f'.\\test{counter}.html')
webbrowser.get(chromepath).open(f'.\\test{counter}.html', new=new)
counter = counter+1

fig = px.scatter_3d(df_scores, x='PC1', y='PC2', z='PC3',
                    color='Species',
                    symbol='Species',
                    opacity=0.5)

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
loadings_label = df_loadings.index
# loadings_label = df_loadings.index.str.strip(' (cm)')

fig = px.scatter_3d(df_loadings, x='PC1', y='PC2', z='PC3',
                    text=loadings_label)

fig.write_html(f'.\\test{counter}.html')
webbrowser.get(chromepath).open(f'.\\test{counter}.html', new=new)
counter = counter+1
