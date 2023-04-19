# Using the linear CV model
from sklearn.linear_model import LassoCV
import App
from sklearn.model_selection import KFold

# Lasso Cross validation
k_folds = KFold(n_splits=5)
lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], cv=k_folds, random_state=0).fit(App.x_train, App.y_train)

# score
print("\nLasso Model............................................\n")
print("The train score for lasso model is", lasso_cv.score(App.x_train, App.y_train))
print("The test score for lasso model is", lasso_cv.score(App.x_test, App.y_test))
