# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas, numpy, matplotlib, and load the Placement Data CSV.  
2. Drop 'sl_no' and 'salary' columns, and check dataset info.  
3. Convert categorical columns to category type and then encode them numerically.  
4. Separate features (`x`) and target (`y`) from the dataset.  
5. Split the data into training and testing sets.  
6. Create a logistic regression model and train it on the training data.  
7. Predict on the test data and evaluate accuracy and confusion matrix.  
8. Predict the output for two new custom input samples.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: V RAKSHITA
RegisterNumber:  212224100049
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset.info()
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary', axis=1)
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
dataset.info()
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
print(x_train.shape)
print(y_train.shape)
from sklearn.linear_model import LogisticRegression
cl=LogisticRegression(max_iter=1000)
cl.fit(x_train,y_train)
y_pred=cl.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,y_test))
confusion_matrix(y_pred,y_test)
cl.predict([[0,87,0,95,0,2,8,0,0,1,5,6]])
cl.predict([[1,2,3,4,5,6,7,8,9,10,11,12]])
```

## Output:

![Screenshot (278)](https://github.com/user-attachments/assets/3c051f25-262f-4c66-9e54-a3729200c730)
![Screenshot (279)](https://github.com/user-attachments/assets/3e8f7363-a58a-4c65-8104-396f4f83caf6)
![Screenshot (280)](https://github.com/user-attachments/assets/90fcb97f-ef62-418a-a569-28ec69623d61)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
