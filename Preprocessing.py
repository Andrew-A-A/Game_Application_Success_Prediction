import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

"""
Global dictionary that will store either mean of feature (for numerical data) 
or mode (for categorical data) in order to fill nulls in testing data with this values """
# Key => column name, Value=>(mean/mode)
global_vars = {}


# Encode specific features and return updated dataframe
def feature_encoder(df, columns):
    for c in columns:
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values))
        df[c] = lbl.transform(list(df[c].values))
    return df


# Scale features in a given range ( a -> b )
def feature_scaling(df, a, b):
    df = np.array(df)
    normalized_x = np.zeros((df.shape[0], df.shape[1]))
    for i in range(df.shape[1]):
        normalized_x[:, i] = ((df[:, i] - min(df[:, i])) / (max(df[:, i]) - min(df[:, i]))) * (b - a) + a
    return normalized_x


# Remove first word of a string that contains some words separated with comma (,)
def remove_first_word(feature):
    feature = list(feature.apply(lambda colm: ', '.join(colm.split(', ')[1:])))
    return feature


# Fill null values with a given value
def fill_nulls(feature, value):
    feature = feature.fillna(value)
    return feature


# Drop given columns names
def drop_columns(df, columns_names):
    for col in columns_names:
        df = df.drop(col, axis=1)
    return df


# Fill nulls in gives feature with the mode of it
def fill_nulls_with_mode(feature):
    mode = feature.mode().iloc[0]
    feature = feature.fillna(mode)
    return feature


""" 
Convert string contains more than one number separated by comma into float numbers
then calculate the sum of them """


def calc_sum_of_list(feature):
    feature = feature.apply(lambda x: sum([float(num.strip(',')) for num in str(x).split()]))
    return feature


# Split specific column that contains date into three features (day,month,year)
def explode_date(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
    df[date_column + '_year'] = df[date_column].dt.year
    df[date_column + '_month'] = df[date_column].dt.month
    df[date_column + '_day'] = df[date_column].dt.day
    df = df.drop(date_column, axis=1)
    return df


# Apply hot one encoding to a given column name
def hot_one_encode(df, column):
    # Convert the string in column Genres into a list
    df[column] = df[column].apply(lambda x: x.replace(' ', '').split(','))
    # Split the values in column Genres into multiple rows
    df = df.explode(column)

    # Apply one-hot encoding to column Genres
    df = pd.get_dummies(df, columns=[column])

    # Drop the duplicates
    df = df.drop_duplicates()
    return df


# Apply cross validation
def cross_validation(model, x_train, y_train):
    k_folds = KFold(n_splits=5)
    scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=k_folds)
    model_score = abs(scores.mean())
    print("model 1 cross validation score is " + str(model_score))


# Detect outliers and replace them with the mean
def outlier_iqr_replace(df):
    # Calculate the IQR for each column
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    # Create a mask for outliers
    outlier_mask = (df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))
    cols_outliers = ["User Rating Count"]
    # Calculate the mean of non-outlier values for each column
    mean = df.loc[:, cols_outliers].mask(outlier_mask).mean()

    # Replace outliers with the mean
    df.loc[:, cols_outliers] = df.loc[:, cols_outliers].mask(outlier_mask, other=int(mean), axis=0)
    return df


def wrapper_feature_selection(df, x_train, y_train, x_test, y_test):
    count = 0
    selected_features = []
    y = df['Average_User_Rating']
    for col in x_train.columns:
        x_current = x_train.loc[:, col]
        linear_model = LinearRegression()
        y_test = np.array(y_test).reshape(-1, 1)
        y_train = np.array(y_train).reshape(-1, 1)
        x_current = np.array(x_current).reshape(-1, 1)
        temp = x_test.loc[:, col]
        temp = np.array(temp).reshape(-1, 1)
        linear_model.fit(x_current, y_train)
        if metrics.mean_squared_error(y_train, linear_model.predict(x_current)) <= 0.5716:
            if x_train[col].nunique() > 2:
                mean = x_train[col].mean()
                global_vars[col] = mean
            else:
                mode = x_train[col].mode()
                global_vars[col] = mode
            count += 1
            selected_features.append(col)
            print(f"Train mean sqError {col}{metrics.mean_squared_error(y_train, linear_model.predict(x_current))}\n")
            print(f"Test mean sqError for {col} {metrics.mean_squared_error(y_test, linear_model.predict(temp))} \n")
    global_vars['In-app Purchases'] = 0.0
    print(f"counter = {count}")
    df = df[selected_features]
    df = df.join(y)
    return df


def feature_scale(df):
    column_names = df.columns.tolist()
    standardization = preprocessing.StandardScaler()
    scaled_data = standardization.fit_transform(df)
    df = pd.DataFrame(scaled_data, columns=column_names)
    return df

# from pandas_profiling import ProfileReport
# generate the profile report
# profile = df.profile_report()

# save the report as a html file
# profile.to_file(output_file='after_removingOutlier.html')

# # Feature Selection
# # Get the correlation between the features
# corr = df.corr()
# # Top 50% Correlation training features with the Value
# top_feature = corr.index[abs(corr['Average_User_Rating']) > 0.1]
# # Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = df[top_feature].corr()
# # df = df[top_feature]
# sns.heatmap(top_corr, annot=True)
# plt.show()
# top_feature = top_feature.delete(-1)


# # Split data into groups
# groups = []
# for col in df.columns:
#     if col.startswith('Genres_') or col == categorical_columns[0] or col == categorical_columns[1] or col == \
#             categorical_columns[2]:
#         group = df[df[col] == 1]['Average_User_Rating']
#         groups.append(group)
# # Perform one-way ANOVA test
# f_statistic, p_value = f_oneway(*groups)
#
# # Print results
# print(f'F-statistic: {f_statistic:.4f}')
# print(f'p-value: {p_value:.4f}')


# temp = X_num.join(y_train)
# # Compute Spearman rank correlation coefficient
# newDf = temp.corr(method="pearson")
#
# # Print the correlation coefficient and p-value
# print("Spearman's rank correlation coefficient: \n", newDf)
# plt.subplots(figsize=(12, 8))
# sns.heatmap(newDf, annot=True)
# plt.show()

# print(df.dtypes)
