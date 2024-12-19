from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn import datasets

X,y=datasets.make_regression(n_samples=1000,n_features=1,noise=20,random_state=4)
X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=200,shuffle=True)

model=LinearRegression(lr=0.1)

model.fit(X_train,y_train)
pred=model.predict(X_test)
def mean_square_error(pred,y_test):
    return np.mean((y_test-pred)**2)
mse=mean_square_error(pred,y_test)
print(mse)