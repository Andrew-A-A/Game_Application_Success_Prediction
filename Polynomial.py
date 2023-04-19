from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from App import *

poly_features = PolynomialFeatures(degree=2)

X_train_poly = poly_features.fit_transform(x_train)

poly_model = linear_model.LinearRegression()
cross_validation(poly_model, X_train_poly, y_train)

poly_model.fit(X_train_poly, y_train)

y_train_predicted = poly_model.predict(X_train_poly)
y_predict = poly_model.predict(poly_features.transform(x_test))

print('Mean Square Error', metrics.mean_squared_error(y_test, y_predict))
