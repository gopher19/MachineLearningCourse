import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')
X = data.iloc[0:100,1:4] #when I use all predictors R square value: 0.46, but when I use only x1,x2,x3 r square value increase to 0.67
y = data.iloc[0:100,-1]
X = X.values
y = y.values
y = y.reshape(100,1)
x3 = data[['x3']].iloc[0:100].values
realtestX = data.iloc[100:,1:6].values

sc1 = StandardScaler()
X = sc1.fit_transform(X)
sc2 = StandardScaler()
y = sc2.fit_transform(y)

list1 = [] #will contain y test values
list2 = [] #will contain y predictions
reg = SVR(kernel = 'rbf')
kf = KFold(n_splits=10)

for train, test in kf.split(X):
    Xtrain,Xtest = X[train],X[test]
    Ytrain,Ytest = y[train],y[test]
    reg.fit(Xtrain,Ytrain)
    yhat = reg.predict(Xtest)
    for i in range(0,len(yhat)):
        list1.append(Ytest[i])
        list2.append(yhat[i])

plt.scatter(x3,list1,color="orange") #y values
plt.scatter(x3,list2,color="purple") #predictions
#plt.plot(x3,list2) # plot of predicted values

plt.xlabel("x3 values")
plt.ylabel("y and prediction values")
plt.title("Actual Values(orange) and Predictions(purple)")
plt.show()
print("R score : ", r2_score(list1,list2))
print("Mean squared error : ",mean_squared_error(list1,list2))
