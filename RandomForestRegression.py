import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('data.csv')
X = data.iloc[0:100,1:6].values
y = data.iloc[0:100,-1].values
x3 = data[['x3']].iloc[0:100].values
realtestX = data.iloc[100:,1:6].values

list1 = [] #will contain y test values
list2 = [] #will contain y predictions
reg = RandomForestRegressor(n_estimators=10,random_state=0)
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
plt.title("Actual value(orange) vs predictions(purple)")
plt.show()
print("R score : ", r2_score(list1,list2))
print("Mean squared error : ",mean_squared_error(list1,list2))

reg.fit(X[:100],y[:100])
estimations = reg.predict(realtestX)
data['Y'][100:] = estimations
data.to_csv('data_with_predictions.csv')
estimations = estimations.reshape(-1,1)
print("Estimations \n",estimations)
