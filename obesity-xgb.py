import numpy as np
from pandas import read_csv
import xgb_test as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data = read_csv('C:\\Users\GUSTAVOJEREZ\PycharmProjects\XGBOOST\obesity.csv', header=None)
dataset = data.values

X = dataset[:, :-1].astype(str)
Y = dataset[:, -1]

encoded_x = None
for i in range(8, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
	feature = onehot_encoder.fit_transform(feature)
	if encoded_x is None:
		encoded_x = feature
	else:
		encoded_x = np.concatenate((encoded_x, feature), axis=1)

label_encoder = LabelEncoder()
label_encoded_y = label_encoder.fit_transform(Y)

X = np.concatenate((X[:,:8],encoded_x),axis=1)

print('Input X', encoded_x.shape)
print(encoded_x[:5, :])
print('Output', label_encoded_y.shape)
print(label_encoded_y[:5])

print('\n\n')
#TRAINING PHASE
xtrain, xtest, ytrain, ytest=train_test_split(encoded_x, label_encoded_y, test_size=0.2, random_state=7)

xgbr = xgb.XGBRegressor(verbosity=0)

#print(xgbr)

xgbr.fit(xtrain, ytrain)

score = xgbr.score(xtrain, ytrain)
print("Training score: ", score)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(xgbr, xtrain, ytrain, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

#PREDICTION PHASE
ypred = xgbr.predict(xtest)

mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

predictions = [round(value) for value in ypred]
# evaluate predictions
accuracy = accuracy_score(ytest, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('------------------------------------\n')
print('test ->', ytest)
print('pred ->', ypred)

x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, predictions, label="predicted")
plt.title("Obesity levels test and predicted data")
plt.legend()
#plt.show()

fig, ax = plt.subplots()
ax.scatter(ytest, predictions)
ax.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()