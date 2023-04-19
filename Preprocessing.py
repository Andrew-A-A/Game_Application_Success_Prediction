import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

global_vars = {}

def feature_encoder(x, colus):
    for c in colus:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x


def feature_scaling(x, a, b):
    x = np.array(x)
    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i] - min(x[:, i])) / (max(x[:, i]) - min(x[:, i]))) * (b - a) + a
    return normalized_x


def remove_first_word(x):
    x = list(x.apply(lambda x: ', '.join(x.split(', ')[1:])))
    return x


def fill_nulls(x, val):
    x = x.fillna(val)
    return x


def drop_columns(x, lst):
    for col in lst:
        x = x.drop(col, axis=1)
    return x


def fill_nulls_with_mode(x):
    mode = x.mode().iloc[0]
    x = x.fillna(mode)
    return x


def calc_sum_of_list(m):
    m = m.apply(lambda x: sum([float(num.strip(',')) for num in str(x).split()]))
    return m


def explode_date(x, column):
    x[column] = pd.to_datetime(x[column], dayfirst=True)
    x[column + '_year'] = x[column].dt.year
    x[column + '_month'] = x[column].dt.month
    x[column + '_day'] = x[column].dt.day
    x = x.drop(column, axis=1)
    return x


def hot_one_encode(s, column):
    # Convert the string in column Genres into a list
    s[column] = s[column].apply(lambda x: x.replace(' ', '').split(','))
    # Split the values in column Genres into multiple rows
    s = s.explode(column)

    # Apply one-hot encoding to column Genres
    s = pd.get_dummies(s, columns=[column])

    # Drop the duplicates
    s = s.drop_duplicates()
    return s


def cross_validation(model, x_train, y_train):
    k_folds = KFold(n_splits=5)
    scores = cross_val_score(model, x_train, y_train, scoring='mean_squared_error', cv=k_folds)
    model_score = abs(scores.mean())
    print("model 1 cross validation score is " + str(model_score))


def outlier_IQR_replace(df):
    # Calculate the IQR for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Create a mask for outliers
    outlier_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    cols_outliers = ["User Rating Count"]
    # Calculate the mean of non-outlier values for each column
    mean = df.loc[:, cols_outliers].mask(outlier_mask).mean()

    # Replace outliers with the mean
    df.loc[:, cols_outliers] = df.loc[:, cols_outliers].mask(outlier_mask, other=int(mean), axis=0)
    return df


def wrapper_feature_selection(df, x_train, y_train, x_test, y_test):
    count = 0
    selected_features = []
    Y = df['Average_User_Rating']
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
            print(
                f"Train mean sqError for{col} {metrics.mean_squared_error(y_train, linear_model.predict(x_current))} \n")
            print(f"Test mean sqError for {col} {metrics.mean_squared_error(y_test, linear_model.predict(temp))} \n")
    global_vars['In-app Purchases'] = 0.0
    print(f"counter = {count}")
    df = df[selected_features]
    df = df.join(Y)
    return df


def feature_scale(df):
    column_names = df.columns.tolist()
    standardization = preprocessing.StandardScaler()
    scaled_data = standardization.fit_transform(df)
    df = pd.DataFrame(scaled_data, columns=column_names)
    return df

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
