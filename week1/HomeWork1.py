
import numpy as np
import pandas as pd

#Q1 :What's the version of Numpy?
print("np version is", np.__version__)

#Q2 : What's the version of Pandas?
print("pd version is", pd.__version__)

#Getting the data
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv')
print(df.head())
print(df)

#Q3 : What's the average price of BMW cars in the dataset?
df_BMW = df[df['Make'] == 'BMW']
print('Average price of the BMW:', df_BMW['MSRP'].mean())

#Q4 : Select a subset of cars after year 2015. How many of them have missing values for Engine HP?
df_2015 = df[df['Year'] >= 2015]
print(df_2015['Engine HP'].isnull().sum())

#Q5 : Calculate the average "Engine HP" in the dataset.
#     Use the fillna method and to fill the missing values in "Engine HP" with the mean value from the previous step.
#     Now, calcualte the average of "Engine HP" again.
#     Has it changed?

mean_hp_before = df['Engine HP'].mean()
print('Engine HP with fillna values:', round(mean_hp_before, 2))

df['Engine HP fillna'] = df['Engine HP'].fillna(df['Engine HP'].mean())
mean_hp_after = df['Engine HP fillna'].mean()
print('Engine HP without fillna values:', round(mean_hp_after, 2))

#Q6 : Select all the "Rolls-Royce" cars from the dataset.
#     Select only columns "Engine HP", "Engine Cylinders", "highway MPG".
#     Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 7 rows).
#     Get the underlying NumPy array. Let's call it X.
#     Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
#     Invert XTX.
#     What's the sum of all the elements of the result?

X = np.array(df[df.Make == "Rolls-Royce"][["Engine HP", "Engine Cylinders", "highway MPG"]].drop_duplicates())
print(X)
print(X.shape)

XTX = X.T.dot(X)
print(XTX)

XTX_invert = np.linalg.inv(XTX)
print(XTX_invert.sum())

#Q7 : Create an array y with values [1000, 1100, 900, 1200, 1000, 850, 1300].
#     Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
#     What's the value of the first element of w?.
y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
w = (XTX_invert.dot(X.T)).dot(y)
print(w[0])
