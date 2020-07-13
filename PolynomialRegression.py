import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('data.csv')
X = data.iloc[0:100,1:4].values
y = data.iloc[0:100,-1].values
x3 = data[['x3']].iloc[0:100].values

list1 = [] #will contain y test values
list2 = [] #will contain y predictions
polyreg = PolynomialFeatures(degree=3)
linreg = LinearRegression()
kf = KFold(n_splits=10)

for train, test in kf.split(X):
    Xtrain,Xtest = X[train],X[test]
    Ytrain,Ytest = y[train],y[test]
    x_polyreg = polyreg.fit_transform(Xtrain)
    linreg.fit(x_polyreg,Ytrain)
    yhat = linreg.predict(polyreg.fit_transform(Xtest))
    for i in range(0,len(yhat)):
        list1.append(Ytest[i])
        list2.append(yhat[i])

plt.scatter(x3,list1,color="orange")
plt.scatter(x3,list2,color="purple")
#plt.plot(x3,list2) # plot of predicted values
plt.xlabel("x3 values")
plt.ylabel("y and prediction values")
plt.title("ActualValues(orange),Predictions(purple)")
plt.show()
print("R score : ", r2_score(list1,list2))
print("Mean squared error : ",mean_squared_error(list1,list2))
