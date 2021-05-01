import numpy
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data = read_csv('C:\\Users\GUSTAVOJEREZ\PycharmProjects\XGBOOST\data_obesity.csv', header=None)
dataset = data.values

X = dataset[:, :-1].astype(str)
Y = dataset[:, -1]

ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X[:,8:-1])

onehot_encoder = OneHotEncoder(sparse=False)
X[:,-1] = X[:,-1].reshape(X.shape[0], 1)
transp = onehot_encoder.fit_transform(X[:,-1])
X = numpy.concatenate((X, transp), axis=1)
# ordinal encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(Y)
# summarize the transformed data

print('Input', X.shape)
print(X[:5, :])
print('Output', y.shape)
print(y[:5])