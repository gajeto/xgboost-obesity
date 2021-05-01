import numpy as np
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data = read_csv('C:\\Users\GUSTAVOJEREZ\PycharmProjects\XGBOOST\data_obesity.csv', header=None)
dataset = data.values

X1 = dataset[:, :-2].astype(str)
X2 = dataset[:,-2].astype(str)
Y = dataset[:, -1]

genders= ['Male','Female']
yes_no = ['no','yes']
frequency = ['no','Sometimes','Frequently','Always']

ordinal_encoder = OrdinalEncoder(categories=[genders, yes_no,yes_no, frequency,yes_no,yes_no,frequency])
X1 = ordinal_encoder.fit_transform(X1[:,8:])
#print('ord_cat: ', ordinal_encoder.categories_)

onehot_encoder = OneHotEncoder(sparse=False)
X2 = X2.reshape(X1.shape[0],1)
X2 = onehot_encoder.fit_transform(X2)
#print('\none_cat: ',onehot_encoder.categories_)

label_encoder = LabelEncoder()
Y= label_encoder.fit_transform(Y)
#print('\nlabel_cat: ',label_encoder.classes_)

X = np.concatenate((X1,X2),axis=1)

print('Input X1', X1.shape)
print(X1[:5, :])
print('Input X', X.shape)
print(X[:5, :])
print('Output', Y.shape)
print(Y[:5])