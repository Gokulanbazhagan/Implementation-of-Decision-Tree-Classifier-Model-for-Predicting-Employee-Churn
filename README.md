# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Gokularamanan k
RegisterNumber:  212222230040
*/
import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours",
        "time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

## 1.Head:
![Screenshot 2024-04-02 162411](https://github.com/HEMAKESHG/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870552/618a62bc-9c2c-4395-8bfe-7c4770ae8b08)
## 2.Accuracy:
![Screenshot 2024-04-02 162318](https://github.com/HEMAKESHG/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870552/62cccf9f-ba03-4aa9-b75d-bac1641e0711)
## 3. Predict:
![Screenshot 2024-04-02 162347](https://github.com/HEMAKESHG/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870552/e1946cfd-ef32-4233-ad0f-e4ccd6b5ee8e)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
