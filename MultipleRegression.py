import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv('data.csv')
veri1 = data[['x2']].values
veri2 = data[['x6']].values
for i in range(0,len(veri1)): #so it prints nothing :) it means those are equal
    if(veri1[i]!=veri2[i]):
        print("there is a data that is not match",veri[i])

X = data[['x1','x2','x3']].iloc[0:100].values
y = data[['Y']].iloc[0:100].values
x3 = data[['x3']].iloc[0:100].values
ones = np.ones((len(X),1))
X = np.append(ones,values=X,axis=1)
reg = LinearRegression()
list1 = [] #will contain y test values
list2 = [] #will contain y predictions
kf = KFold(n_splits=10)

for train, test in kf.split(X):
    Xtrain,Xtest = X[train],X[test]
    Ytrain,Ytest = y[train],y[test]
    reg.fit(Xtrain,Ytrain)
    yhat = reg.predict(Xtest)
    for i in range(0,len(yhat)):
        list1.append(Ytest[i][0])
        list2.append(yhat[i][0])

print("R squared error : ", r2_score(list1,list2))
print("Mean squared error : ",mean_squared_error(list1,list2))
plt.scatter(x3,list1,color="orange") #actual values
plt.scatter(x3,list2) #predicted values
#plt.plot(x3,list2) # plot of predicted values
plt.title("Actual Values(Orange), Predictions(Blue)")
plt.xlabel("x3 values")
plt.ylabel("y and predictions")
plt.show()

"""
Backward Elimination process 1

Significance level = 0.05

X = data[['x1','x2','x3','x4','x5']].iloc[0:100].values
y = data[['Y']].iloc[0:100].values
ones = np.ones((len(X),1))
X = np.append(ones,values=X,axis=1)
r_ols = sm.OLS(endog=y,exog=X)
r = r_ols.fit()
print(r.summary())

X5 has the biggest p value so I remove it

Backward Elimination process 2

X = data[['x1','x2','x3','x4']].iloc[0:100].values
y = data[['Y']].iloc[0:100].values
ones = np.ones((len(X),1))
X = np.append(ones,values=X,axis=1)
r_ols = sm.OLS(endog=y,exog=X)
r = r_ols.fit()
print(r.summary())

X4 has 0.10 still above the significance level so I removed it

Backward Elimination process 3

X = data[['x1','x2','x3']].iloc[0:100].values
y = data[['Y']].iloc[0:100].values
ones = np.ones((len(X),1))
X = np.append(ones,values=X,axis=1)
r_ols = sm.OLS(endog=y,exog=X)
r = r_ols.fit()
print(r.summary())

now all predictors are below the SL

"""
