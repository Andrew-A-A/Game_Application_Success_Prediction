from App import *
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

# Ridge Regression Model
fold = KFold(n_splits=5)
ridgeReg = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], cv=fold)
ridgeReg.fit(x_train, y_train)

# train and test score for ridge regression
train_score_ridge = ridgeReg.score(x_train, y_train)
test_score_ridge = ridgeReg.score(x_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))
