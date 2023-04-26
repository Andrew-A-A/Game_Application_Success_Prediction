from datetime import datetime

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures

lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words]
    s_words = set(words)
    return s_words


def feature_extraction(description_column):
    returned_list = []
    for description in description_column:
        words = preprocess_text(description)
        meaningful_words = []
        for word in words:
            if len(word) < 3:
                continue
            synsets = wordnet.synsets(word)
            if synsets:
                meaningful_words.append(word)

        pos_tags = nltk.pos_tag(meaningful_words)
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        nouns = ' '.join(nouns)
        returned_list.append(nouns)
    returned_list = pd.DataFrame({'New': returned_list})
    # print(returned_list)
    features = vectorizer.fit_transform(returned_list['New'])
    print(features)
    # Calculate the average TF-IDF value for each row
    max_tfidf = features.max(axis=1)
    max_tfidf = max_tfidf.todense().A1
    # Assign the average TF-IDF values to a new column in the data frame
    description_column = max_tfidf
    return description_column


def drop_columns(data_frame, columns_names):
    for COL in columns_names:
        data_frame = data_frame.drop(COL, axis=1)
    return data_frame


def fill_nulls(feature, value):
    feature = feature.fillna(value)
    return feature


def fill_nulls_with_mode(feature):
    mode = feature.mode().iloc[0]
    feature = feature.fillna(mode)
    return feature


def remove_first_word(feature):
    feature = list(
        feature.apply(lambda colm: ', '.join(colm.split(', ')[1:] if len(colm.split(', ')) > 1 else colm.split(', '))))
    return feature


def remove_special_chars(data_frame, column_name):
    # Define a pattern to match special characters
    # pattern = r'[^a-zA-Z0-9\s]'
    pattern = r'[^a-zA-Z0-9\s.,:\'()\-"\\]'
    # Create a boolean mask to identify rows with special characters in the specified column
    mask = data_frame[column_name].str.contains(pattern)

    # # Print the rows that will be deleted
    # print("Rows to be deleted:")
    # print(df[mask])
    # Drop rows with special characters in the specified column
    data_frame = data_frame[~mask]
    return data_frame


def weight_genres(genres):
    # Create a dictionary to hold the weights
    weights = {}
    # Loop through the genres list and assign weights based on order of appearance
    for i, genre in enumerate(genres):
        weights[genre] = len(genres) - i
    return weights


def calc_sum_of_list(feature):
    feature = feature.apply(lambda x: sum([float(num.strip(',')) for num in str(x).split()]))
    return feature


def wrapper_feature_selection(x, y):
    linear_model = LinearRegression()
    rfe = RFE(estimator=linear_model, n_features_to_select=15)
    rfe.fit(x, y)
    # Print the selected features
    print(x.columns[rfe.support_])
    x_train_selected = x[x.columns[rfe.support_]]
    return x_train_selected


def get_mode_or_mean(feature, data_frame, flag=0):
    if flag == 0:
        return data_frame[feature].mode().iloc[0]
    else:
        return data_frame[feature].mean()


global_vars = {}
# Load the csv file
df = pd.read_csv("games-regression-dataset.csv")
print(df.dtypes)
# Split data frame to X and Y
Y = df['Average_User_Rating']
X = df.drop('Average_User_Rating', axis=1)
print(X.columns)

# X['Description'] = vectorizer.fit_transform(X['Description']).shape[1]

# Split the X and the Y to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)

# ----------------------------------------Training Preprocessing----------------------------------------
# Drop unimportant columns
unimportant_columns = ['URL', 'ID', 'Name', 'Subtitle', 'Icon URL']
x_train = drop_columns(x_train, unimportant_columns)

# x_train['Description'] = x_train['Description'].apply(lambda x: feature_extraction(x).shape[1])
x_train['Description'] = feature_extraction(x_train['Description'])

# Fill missing values in "In-app purchases" column with zero
x_train['In-app Purchases'] = fill_nulls(x_train['In-app Purchases'], 0)
global_vars['In-app Purchase'] = 0
x_train['In-app Purchases'] = calc_sum_of_list(x_train['In-app Purchases'])
# Fill missing values in column 'Languages' with the mode
global_vars['Languages'] = x_train['Languages'].mode().iloc[0]
x_train['Languages'] = fill_nulls_with_mode(x_train['Languages'])

# change datatypes from object
x_train = x_train.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
x_train['Age Rating'] = x_train['Age Rating'].str.replace('+', '', regex=False)
# Convert the 'Age rating' column to an integer data type
x_train['Age Rating'] = x_train['Age Rating'].astype(int)
global_vars['Age Rating'] = x_train['Age Rating'].mode().iloc[0]

global_vars['Primary Genre'] = x_train['Primary Genre'].mode().iloc[0]
# Remove the primary genre from the "Genres" feature
x_train['Primary Genre'].fillna(global_vars['Primary Genre'], inplace=True)
x_train['Genres'].fillna(global_vars['Primary Genre'], inplace=True)
x_train['Genres'] = remove_first_word(x_train['Genres'])
x_train['Genres'] = x_train['Genres'].apply(lambda x: x.replace(' ', '').split(','))

# Encode categorical columns
categorical_columns = ('Developer', 'Languages', 'Primary Genre')
# feature_encoder_fit(x_train, categorical_columns)
print(x_train.shape)
data = x_train.join(y_train)
data = remove_special_chars(data, 'Developer')
y_train = data['Average_User_Rating']
x_train = data.drop('Average_User_Rating', axis=1)
x_train['Developer'] = x_train['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)
# filling Developer with the mentioned developer in the description column
global_vars['Developer'] = 'Unknown'
global_vars['User Rating Count'] = x_train['User Rating Count'].mean()
global_vars['Size'] = x_train['Size'].mean()
# pattern = r'(\\u[0-9a-fA-F]{4})+'
# x_train['Developer'].dropna(axis=0,inplace= True)
# print(x_train.shape)

dev_encoder = LabelEncoder()
x_train['Developer'] = dev_encoder.fit_transform(x_train['Developer'])

lang_encoder = LabelEncoder()
x_train['Languages'] = lang_encoder.fit_transform(x_train['Languages'])

primary_genre_encoder = LabelEncoder()
x_train['Primary Genre'] = primary_genre_encoder.fit_transform(x_train['Primary Genre'])
x_train.drop('Primary Genre', axis=1, inplace=True)

x_train['Original Release Date'] = pd.to_datetime(x_train['Original Release Date'], errors='coerce', format='%d/%m/%Y')
x_train['Current Version Release Date'] = pd.to_datetime(x_train['Current Version Release Date'], errors='coerce',
                                                         format='%d/%m/%Y')
x_train['Difference in Days'] = (x_train['Current Version Release Date'] - x_train['Original Release Date']).dt.days

global_vars['Original Release Date'] = datetime.now()
global_vars['Current Version Release Date'] = datetime.now()

print(x_train.shape)
x_train.drop(['Original Release Date', 'Current Version Release Date'], axis=1, inplace=True)
# Apply the weight_genres function to the genres column and store the results in a new column called genre_weights
x_train['genre_weights'] = x_train['Genres'].apply(weight_genres)

# Create a list of all unique genres in the dataset
unique_genres = list(set([genre for genres in x_train['Genres'] for genre in genres]))

# Create a binary column for each unique genre and set the value to the weight assigned by the weight_genres function
for genre in unique_genres:
    x_train[genre] = x_train['genre_weights'].map(lambda x: x.get(genre, 0))

# Drop the genre_weights column since it is no longer needed
x_train.drop('genre_weights', axis=1, inplace=True)
print(x_train.shape)

data = x_train.join(y_train)
game_data = data.iloc[:, :]
corr = game_data.corr(method='spearman', numeric_only=True)
# #Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Average_User_Rating']) > 0.03]
print('Top Features', top_feature)
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = game_data[top_feature].corr(method='spearman')
sns.heatmap(top_corr, annot=True)
# plt.show()

x_data = game_data[top_feature]
print(x_data.columns)

standardization = StandardScaler()
game_data = standardization.fit_transform(x_data)
game_data = pd.DataFrame(game_data, columns=top_feature)
y_train = game_data['Average_User_Rating']
x_train = game_data.drop('Average_User_Rating', axis=1)

# ---------------------------------Testing Preprocessing-----------------------------------

x_test = drop_columns(x_test, unimportant_columns)

# x_test['Description'] = x_test['Description'].apply(lambda x: feature_extraction(x).shape[1])
x_test['Description'] = feature_extraction(x_test['Description'])

# Fill missing values in "In-app purchases" column with zero
# x_test['In-app Purchases'] = fill_nulls(x_test['In-app Purchases'], 0)
x_test['In-app Purchases'] = calc_sum_of_list(x_test['In-app Purchases'])
x_test['Original Release Date'] = pd.to_datetime(x_test['Original Release Date'], errors='coerce', format='%d/%m/%Y')
x_test['Current Version Release Date'] = pd.to_datetime(x_test['Current Version Release Date'], errors='coerce',
                                                        format='%d/%m/%Y')

# Remove the '+' sign from the 'Age rating' column
x_test['Age Rating'] = x_test['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
x_test['Age Rating'] = x_test['Age Rating'].astype(int)

# Fill missing values in column 'Languages' with the mode
# x_test['Languages'] = fill_nulls_with_mode(x_train['Languages'])
print(x_train.dtypes)
global_vars['Genres'] = global_vars['Primary Genre']
for col in x_test.columns:
    if col == 'In-app Purchases' or col == 'Price' or col == 'Description':
        x_test[col] = fill_nulls(x_test[col], 0)
    else:
        x_test[col].fillna(global_vars[col], inplace=True)

x_test['Difference in Days'] = (x_test['Current Version Release Date'] - x_test['Original Release Date']).dt.days
x_test.drop(['Original Release Date', 'Current Version Release Date'], axis=1, inplace=True)
# change datatypes from object
x_test = x_test.convert_dtypes()

# Remove the primary genre from the "Genres" feature
x_test['Genres'] = remove_first_word(x_test['Genres'])
x_test['Genres'] = x_test['Genres'].apply(lambda x: x.replace(' ', '').split(','))

# Apply the weight_genres function to the genres column and store the results in a new column called genre_weights
x_test['genre_weights'] = x_test['Genres'].apply(weight_genres)

# Apply one-hot encoding to the 'Genres' column
one_hot_test = x_test['Genres'].str.get_dummies(',')

# Add missing columns to the one-hot encoded test data
missing_cols = set(x_train.columns) - set(one_hot_test.columns)
for col in missing_cols:
    one_hot_test[col] = 0
# Sort the columns in the test data in the same order as in the training data
one_hot_test = one_hot_test[x_train.columns]

# Apply the weighted one-hot encoding to the test data
for genre in unique_genres:
    x_test[genre] = x_test['genre_weights'].map(lambda x: x.get(genre, 0))

x_test.drop('genre_weights', axis=1, inplace=True)
x_test_data = x_test.join(y_test)
col_test = x_test_data.columns
x_test_data = standardization.transform(x_test_data[top_feature])
x_test_data = pd.DataFrame(x_test_data, columns=top_feature)

y_test = x_test_data['Average_User_Rating']
x_test = x_test_data.drop('Average_User_Rating', axis=1)
# ----------------------------------------------------Models----------------------------------------------------------
print("\nPolynomial Regression Model............................\n")
poly_features = PolynomialFeatures(degree=4)

X_train_poly = poly_features.fit_transform(x_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_train_predicted = poly_model.predict(X_train_poly)
y_predict = poly_model.predict(poly_features.transform(x_test))
print('Mean Square Error Train', mean_squared_error(y_train, y_train_predicted))
print('Mean Square Error Test', mean_squared_error(y_test, y_predict))

k_folds = KFold(n_splits=15)
scores = cross_val_score(poly_model, x_train, y_train, scoring='neg_mean_squared_error', cv=k_folds)
model_score = abs(scores.mean())
print("model cross validation score is " + str(model_score))

print("\nLinear Regression Model...............................\n")
linearReg = LinearRegression()
linearReg.fit(x_train, y_train)

y_train_predicted = linearReg.predict(x_train)
y_predict = linearReg.predict(x_test)
print('Mean Square Error Train', mean_squared_error(y_train, y_train_predicted))
print('Mean Square Error Test', mean_squared_error(y_test, y_predict))

k_folds = KFold(n_splits=15)
scores = cross_val_score(linearReg, x_train, y_train, scoring='neg_mean_squared_error', cv=k_folds)
model_score = abs(scores.mean())
print("model cross validation score is " + str(model_score))

# Lasso Cross validation
k_folds = KFold(n_splits=15)
lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], cv=k_folds, random_state=0).fit(x_train, y_train)

# score
print("\nLasso Model............................................\n")
print("The train score for lasso model is", lasso_cv.score(x_train, y_train))
print("The test score for lasso model is", lasso_cv.score(x_test, y_test))

fold = KFold(n_splits=15)
ridgeReg = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], cv=fold)
ridgeReg.fit(x_train, y_train)
# train and test score for ridge regression
train_score_ridge = ridgeReg.score(x_train, y_train)
test_score_ridge = ridgeReg.score(x_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))

print("\nElastic Net Model........................................\n")
en = ElasticNet()

parameters = {'alpha': [0.1, 0.5, 1, 5, 10],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

grid_search = GridSearchCV(estimator=en, param_grid=parameters,
                           scoring='neg_mean_squared_error', cv=10)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print('Best Hyperparameters:', best_params)

best_score = np.sqrt(-grid_search.best_score_)
print('Best RMSE:', best_score)

en_best = ElasticNet(**best_params)
en_best.fit(x_train, y_train)

y_pred = en_best.predict(x_test)
mse_train = mean_squared_error(y_train, en_best.predict(x_train))
mse_test = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error of Train:', mse_train)
print('Mean Squared Error of Test:', mse_test)
print('R2 Score:', r2)
