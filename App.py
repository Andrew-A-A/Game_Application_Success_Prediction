import pickle
from Test import *
import pandas as pd

models = pickle.load(open('models.pkl', 'rb'))
ploy_model = models['polynomial']
en_best = models['elastic net']
lasso_cv = models['lassoCv']
linearReg = models['linear regression']
ridgeReg = models['ridgeCv']
test_data = pd.read_csv("test.csv")

variables = pickle.load(open('vars.pkl', 'rb'))
unimportant_columns = variables['unimportant columns']
primary_genre_encode = variables['primary genre encoder']
top_feature = variables['top feature']
x_train = variables['x train']
unique_genres = variables['unique genres']
standardization = variables['standardization']
dev_encoder = variables['dev encoder']
lang_encoder = variables['lang encoder']
global_vars = variables['global variables']

Y = test_data['Average User Rating']
X = test_data.drop('Average User Rating', axis=1)
X, Y = preprocess_test_data(X, Y, unimportant_columns, global_vars, dev_encoder, lang_encoder, primary_genre_encode,
                            top_feature, x_train, unique_genres, standardization)

print(Y)
