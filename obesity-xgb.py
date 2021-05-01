import numpy as np
from pandas import read_csv
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data = read_csv('C:\\Users\GUSTAVOJEREZ\PycharmProjects\XGBOOST\obesity.csv', header=None)
dataset = data.values

X1 = dataset[:, :-2].astype(str)
X2 = dataset[:,-2].astype(str)
Y = dataset[:, -1]

genders= ['Male','Female']
yes_no = ['no','yes']
frequency = ['no','Sometimes','Frequently','Always']

ordinal_encoder = OrdinalEncoder(categories=[genders, yes_no,yes_no, frequency,yes_no,yes_no,frequency])
X1 = ordinal_encoder.fit_transform(X1[:,8:])
print('ord_cat: ', ordinal_encoder.categories_)

onehot_encoder = OneHotEncoder(sparse=False)
X2 = X2.reshape(X1.shape[0],1)
X2 = onehot_encoder.fit_transform(X2)
#print('\none_cat: ',onehot_encoder.categories_)

label_encoder = LabelEncoder()
Y= label_encoder.fit_transform(Y)
#print('\nlabel_cat: ',label_encoder.classes_)

X = np.concatenate((X1,X2),axis=1)

xtrain, xtest, ytrain, ytest=train_test_split(X, Y, test_size=0.3)

xgbr = xgb.XGBRegressor(verbosity=0)

xgbr.fit(xtrain, ytrain)

score = xgbr.score(xtrain, ytrain)
print("Training score: ", score)

scores = cross_val_score(xgbr, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, xtrain, ytrain, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

ypred = xgbr.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, ypred, label="predicted")
plt.title("Obesity levels test and predicted data")
plt.legend()
plt.show()
