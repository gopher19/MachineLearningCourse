import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('data.csv')
X = data.iloc[0:100,1:6].values
y = data.iloc[0:100,-1].values
x3 = data[['x3']].iloc[0:100].values

list1 = [] #will contain y test values
list2 = [] #will contain y predictions
reg = DecisionTreeRegressor(max_depth=10,random_state=0)
kf = KFold(n_splits=10)

for train, test in kf.split(X):
    Xtrain,Xtest = X[train],X[test]
    Ytrain,Ytest = y[train],y[test]
    reg.fit(Xtrain,Ytrain)
    yhat = reg.predict(Xtest)
    for i in range(0,len(yhat)):
        list1.append(Ytest[i])
        list2.append(yhat[i])

plt.scatter(x3,list1,color="orange")
plt.scatter(x3,list2,color="black") # plot of predicted values
plt.xlabel("x3 values")
plt.ylabel("y and prediction values")
plt.title("Actual Values(orange), Predictions(black)")
plt.show()

print("R score : ", r2_score(list1,list2))
print("Mean squared error : ",mean_squared_error(list1,list2))
