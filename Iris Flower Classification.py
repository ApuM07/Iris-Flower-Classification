#Oasis Infobytes_Data Science Internship
#Task 1 : IRIS FLOWER CLASSIFICATION
#Name of Intern : APU MANDAL
#Batch  : SEPTEMBER Phase 1 AICTE OIB-SIP 2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

dt=pd.read_csv("Iris.csv")
print("Data In CSV File:")
print(dt)

print("Information of Data:")
dt.info()
print("Structure of Data:")
print(dt.describe())

dt=dt.drop(columns="Id")
print("Value Of Species:")
val=dt['Species'].value_counts()
print(val)

sns.countplot(x='Species',data=dt)
plt.show()

x=dt.iloc[:,:4]
y=dt.iloc[:,4]
print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)*100

print("\nAccuracy of the model: {:.2f}\n".format(accuracy))

print("T H A N K   Y O U")
