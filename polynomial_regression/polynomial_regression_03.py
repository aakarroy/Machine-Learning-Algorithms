"""Importing Libraries"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

"""Getting raw data"""
raw_data = pd.read_csv(r"manufacturing.csv")
X = raw_data.drop("Quality Rating",axis=1).values
Y = raw_data["Quality Rating"]

"""Pipeline"""
p = Pipeline([
    ("scaler",StandardScaler()),
    ("poly",PolynomialFeatures(degree=2)),
    ("model",LinearRegression())
    ])
"""K-Fold Setup"""
y_pred = cross_val_predict(p,X,Y,cv=3)
print(r2_score(Y,y_pred))

"""Plotting"""
plt.scatter(Y,y_pred)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], '--', color='red', lw=2, label='Perfect Prediction')
plt.show()

