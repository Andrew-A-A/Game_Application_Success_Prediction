import numpy
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import seaborn as sns
from sklearn import metrics
from datetime import datetime
def drop_columns(df, columns_names):
    for col in columns_names:
        df = df.drop(col, axis=1)
    return df

def fill_nulls(feature, value):
    feature = feature.fillna(value)
    return feature

def fill_nulls_with_mode(feature):
    mode = feature.mode().iloc[0]
    feature = feature.fillna(mode)
    return feature


def remove_first_word(feature):
    feature = list(feature.apply(lambda colm: ', '.join(colm.split(', ')[1:] if len(colm.split(', '))>1 else colm.split(', '))))
    return feature


def remove_special_chars(df, column_name):
    # Define a pattern to match special characters
    # pattern = r'[^a-zA-Z0-9\s]'
    pattern = r'[^a-zA-Z0-9\s.,:\'()\-"\\]'
    # Create a boolean mask to identify rows with special characters in the specified column
    mask = df[column_name].str.contains(pattern)

    # # Print the rows that will be deleted
    # print("Rows to be deleted:")
    # print(df[mask])
    # Drop rows with special characters in the specified column
    df = df[~mask]
    return df

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


def wrapper_feature_selection(x_train, y_train):
    linear_model=LinearRegression()
    rfe=RFE(estimator=linear_model,n_features_to_select=15)
    rfe.fit(x_train,y_train)
    # Print the selected features
    print(x_train.columns[rfe.support_])
    x_train_selected = x_train[x_train.columns[rfe.support_]]
    return x_train_selected

def get_mode_or_mean(feature,df,flag=0):
    if flag==0:
        return df[feature].mode().iloc[0]
    else:
        return df[feature].mean()

global_vars = {}
# Load the csv file
df = pd.read_csv("games-regression-dataset.csv")
print(df.dtypes)
# Split data frame to X and Y
Y = df['Average_User_Rating']
X = df.drop('Average_User_Rating', axis=1)
print(X.columns)
# Split the X and the Y to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=0)

#----------------------------------------Training Preprocessing----------------------------------------
# Drop unimportant columns
unimportant_columns = ['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'Description']
x_train = drop_columns(x_train, unimportant_columns)

# Fill missing values in "In-app purchases" column with zero
x_train['In-app Purchases'] = fill_nulls(x_train['In-app Purchases'], 0)
global_vars['In-app Purchase']=0
x_train['In-app Purchases'] = calc_sum_of_list(x_train['In-app Purchases'])
# Fill missing values in column 'Languages' with the mode
global_vars['Languages']=x_train['Languages'].mode().iloc[0]
x_train['Languages'] = fill_nulls_with_mode(x_train['Languages'])


# change datatypes from object
x_train = x_train.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
x_train['Age Rating'] = x_train['Age Rating'].str.replace('+', '', regex=False)
# Convert the 'Age rating' column to an integer data type
x_train['Age Rating'] = x_train['Age Rating'].astype(int)
global_vars['Age Rating']=x_train['Age Rating'].mode().iloc[0]

global_vars['Primary Genre']=x_train['Primary Genre'].mode().iloc[0]
# Remove the primary genre from the "Genres" feature
x_train['Primary Genre'].fillna(global_vars['Primary Genre'],inplace=True)
x_train['Genres'].fillna(global_vars['Primary Genre'],inplace=True)
x_train['Genres'] = remove_first_word(x_train['Genres'])
x_train['Genres'] = x_train['Genres'].apply(lambda x: x.replace(' ', '').split(','))


# Encode categorical columns
categorical_columns = ('Developer', 'Languages', 'Primary Genre')
# feature_encoder_fit(x_train, categorical_columns)
print(x_train.shape)
data=x_train.join(y_train)
data=remove_special_chars(data,'Developer')
y_train = data['Average_User_Rating']
x_train = data.drop('Average_User_Rating', axis=1)
x_train['Developer'] = x_train['Developer'].str.replace(r'\\xe7\\xe3o', ' ')
global_vars['Developer']=x_train['Developer'].mode().iloc[0]
global_vars['User Rating Count']=x_train['User Rating Count'].mean()
global_vars['Size']=x_train['Size'].mean()
# pattern = r'(\\u[0-9a-fA-F]{4})+'
# x_train['Developer'].dropna(axis=0,inplace= True)
# print(x_train.shape)

dev_encoder=LabelEncoder()
x_train['Developer']=dev_encoder.fit_transform(x_train['Developer'])

lang_encoder=LabelEncoder()
x_train['Languages'] = lang_encoder.fit_transform(x_train['Languages'])

primary_genre_encoder=LabelEncoder()
# x_train['Primary Genre']= primary_genre_encoder.fit_transform(x_train['Primary Genre'])
x_train.drop('Primary Genre',axis=1,inplace=True)
x_train['Original Release Date']=pd.to_datetime(x_train['Original Release Date'], errors = 'coerce')
x_train['Current Version Release Date']=pd.to_datetime(x_train['Current Version Release Date'], errors = 'coerce')
x_train['Difference in Days']=(x_train['Current Version Release Date']-x_train['Original Release Date']).dt.days

global_vars['Original Release Date']=datetime.now()
global_vars['Current Version Release Date']=datetime.now()


print(x_train.shape)
x_train.drop(['Original Release Date','Current Version Release Date'],axis=1,inplace=True)
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

data=x_train.join(y_train)
game_data = data.iloc[:,:]
corr = game_data.corr(method='spearman')
# #Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Average_User_Rating'])>0.03]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = game_data[top_feature].corr(method='spearman')
sns.heatmap(top_corr, annot=True)
# plt.show()

x_data=game_data[top_feature]
print(x_data.columns)

standardization= StandardScaler()
game_data=standardization.fit_transform(x_data)
game_data=pd.DataFrame(game_data, columns=top_feature)
y_train = game_data['Average_User_Rating']
x_train = game_data.drop('Average_User_Rating', axis=1)


#---------------------------------Testing Preprocessing-----------------------------------

x_test= drop_columns(x_test,unimportant_columns)

# Fill missing values in "In-app purchases" column with zero
# x_test['In-app Purchases'] = fill_nulls(x_test['In-app Purchases'], 0)
x_test['In-app Purchases'] = calc_sum_of_list(x_test['In-app Purchases'])
x_test['Original Release Date']=pd.to_datetime(x_test['Original Release Date'], errors = 'coerce')
x_test['Current Version Release Date']=pd.to_datetime(x_test['Current Version Release Date'], errors = 'coerce')



# Fill missing values in column 'Languages' with the mode
# x_test['Languages'] = fill_nulls_with_mode(x_train['Languages'])
print(x_train.dtypes)
x_test.drop('Primary Genre', axis=1, inplace=True)
for col in x_test.columns:
    if col == 'In-app Purchases' or col=='Genres' or col=='Price':
        x_test[col] = fill_nulls(x_test[col], 0)
    else:
        x_test[col].fillna(global_vars[col],inplace=True)


x_test['Difference in Days']=(x_test['Current Version Release Date']-x_test['Original Release Date']).dt.days
x_test.drop(['Original Release Date','Current Version Release Date'],axis=1,inplace=True)
# change datatypes from object
x_test = x_test.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
x_test['Age Rating'] = x_test['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
x_test['Age Rating'] = x_test['Age Rating'].astype(int)

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
x_test_data=x_test.join(y_test)
col_test=x_test_data.columns
x_test_data=standardization.transform(x_test_data[top_feature])
x_test_data=pd.DataFrame(x_test_data,columns=top_feature)

y_test = x_test_data['Average_User_Rating']
x_test = x_test_data.drop('Average_User_Rating', axis=1)
#----------------------------------------------------Modelssss----------------------------------------------------------
print("\nPolynomial Regression Model............................\n")
poly_features = PolynomialFeatures(degree=2)

X_train_poly = poly_features.fit_transform(x_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_train_predicted = poly_model.predict(X_train_poly)
y_predict = poly_model.predict(poly_features.transform(x_test))
print('Mean Square Error Train', metrics.mean_squared_error(y_train, y_train_predicted))
print('Mean Square Error Test', metrics.mean_squared_error(y_test, y_predict))

k_folds = KFold(n_splits=15)
scores = cross_val_score(poly_model, x_train, y_train, scoring='neg_mean_squared_error', cv=k_folds)
model_score = abs(scores.mean())
print("model cross validation score is " + str(model_score))

print("\nLinear Regression Model...............................\n")
linearReg=LinearRegression()
linearReg.fit(x_train,y_train)

y_train_predicted = linearReg.predict(x_train)
y_predict = linearReg.predict(x_test)
print('Mean Square Error Train', metrics.mean_squared_error(y_train, y_train_predicted))
print('Mean Square Error Test', metrics.mean_squared_error(y_test, y_predict))

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


fold = KFold(n_splits=25)
ridgeReg = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], cv=fold)
ridgeReg.fit(x_train, y_train)
# train and test score for ridge regression
train_score_ridge = ridgeReg.score(x_train, y_train)
test_score_ridge = ridgeReg.score(x_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))