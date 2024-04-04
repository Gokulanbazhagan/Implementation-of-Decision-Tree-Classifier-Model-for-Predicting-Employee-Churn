# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import all the packages that helps to implement Decision Tree.
2.Download and upload required csv file or dataset for predecting Employee Churn.
3.Initialize variables with required features.
4.And implement Decision tree classifier to predict Employee Churn.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Gokularamanan k
RegisterNumber:  212222230049
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor, plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plot_tree(dt,feature_names=x.columns,class_names=['Salary'], filled=True)
plt.show()

```

## Output:

1.Head:

![image](https://github.com/Gokulanbazhagan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119518996/dc694c2a-2d85-4110-b31c-70ecb924a930)

2.Mean Square error:

![image](https://github.com/Gokulanbazhagan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119518996/b1909abb-e101-4b5f-98ec-82ccfc204ea4)

3.Testing of Model:

![image](https://github.com/Gokulanbazhagan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119518996/65d82966-1d4d-4aaf-ab2a-2c98b0f40e74)

4.Decision Tree:

 ![image](https://github.com/Gokulanbazhagan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119518996/e4036864-1a0b-4042-bca7-2f9309fb2dc0)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
