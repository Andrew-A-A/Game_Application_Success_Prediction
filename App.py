import pickle
from sklearn.metrics import mean_squared_error, r2_score
from Test import *

# region Load models and vars
models = pickle.load(open('models.pkl', 'rb'))
poly_model = models['polynomial']
en_best = models['elastic net']
lasso_cv = models['lassoCv']
linearReg = models['linear regression']
ridgeReg = models['ridgeCv']
poly_features = models['poly features']
grid_search = models['grid search']

variables = pickle.load(open('vars.pkl', 'rb'))
unimportant_columns = variables['unimportant columns']
primary_genre_encode = variables['primary genre encoder']
top_feature = variables['top feature']
x_train = variables['x train']
y_train = variables['y train']
unique_genres = variables['unique genres']
standardization = variables['standardization']
dev_encoder = variables['dev encoder']
lang_encoder = variables['lang encoder']
global_vars = variables['global variables']
# endregion
test_data = pd.read_csv("test.csv")
test_data.dropna(inplace=True)
y_test = test_data['Average User Rating']
x_test = test_data.drop('Average User Rating', axis=1)
x_test, y_test = preprocess_test_data(x_test, y_test, unimportant_columns, global_vars, dev_encoder, lang_encoder,
                                      primary_genre_encode, top_feature, x_train, unique_genres, standardization)

# de-standardize y_test
y_true = standardization.inverse_transform(x_test.join(y_test))[:, -1:]

print("\nPolynomial Regression Model............................\n")

X_train_poly = poly_features.transform(x_train)
y_train_predicted = poly_model.predict(X_train_poly)
y_predict = poly_model.predict(poly_features.transform(x_test))
print('Mean Square Error Train', mean_squared_error(y_train, y_train_predicted))
print('Mean Square Error Test', mean_squared_error(y_test, y_predict))

k_folds = KFold(n_splits=15)
scores = cross_val_score(poly_model, x_train, y_train, scoring='neg_mean_squared_error', cv=k_folds)
model_score = abs(scores.mean())
print("model cross validation score is " + str(model_score))

print("\nLinear Regression Model...............................\n")
y_predict_linear = linearReg.predict(x_test)
# convert the numpy array to a DataFrame
y_train_predicted = linearReg.predict(x_train)
y_predict_linear = pd.DataFrame(y_predict_linear)
y_predict_linear = standardization.inverse_transform(x_test.join(y_predict_linear))[:, -1:]
print('Mean Square Error Train', mean_squared_error(y_train, y_train_predicted))
print('Mean Square Error Test', mean_squared_error(y_true, y_predict_linear))

k_folds = KFold(n_splits=15)
scores = cross_val_score(linearReg, x_train, y_train, scoring='neg_mean_squared_error', cv=k_folds)
model_score = abs(scores.mean())
print("model cross validation score is " + str(model_score))

print("\nLasso Model............................................\n")
print("The train score for lasso model is", lasso_cv.score(x_train, y_train))
print("The test score for lasso model is", lasso_cv.score(x_test, y_test))

print("\nRidge Model............................................\n")
# train and test score for ridge regression
train_score_ridge = ridgeReg.score(x_train, y_train)
test_score_ridge = ridgeReg.score(x_test, y_test)
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))


print("\nElastic Net Model........................................\n")

best_params = grid_search.best_params_
print('Best Hyperparameters:', best_params)

best_score = np.sqrt(-grid_search.best_score_)
print('Best RMSE:', best_score)

y_pred = en_best.predict(x_test)
mse_train = mean_squared_error(y_train, en_best.predict(x_train))
mse_test = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error of Train:', mse_train)
print('Mean Squared Error of Test:', mse_test)
print('R2 Score:', r2)
