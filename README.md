# casestudy_lab_1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('diamonds.csv')

data.head()
data.tail()
data.info()
data.describe()
data.corr()
data.isna()                         
data.isna().sum()                 

plt.figure(figsize=(10,8))
sns.pairplot(data)
data.columns

encoder=LabelEncoder()
data['clarity']=encoder.fit_transform(data['clarity'])
x=data.drop(["clarity"],axis=1)
y=data["price"]

data['cut']=encoder.fit_transform(data['cut'])
x=data.drop(["cut"],axis=1)

data['color']=encoder.fit_transform(data['color'])
x=data.drop(["color"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                    test_size=0.20,
                                    random_state=0)

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_
y_pred=regressor.predict(x_test)
y_pred
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
metrics.mean_absolute_error(y_test,y_pred)
metrics.r2_score(y_test,y_pred)
