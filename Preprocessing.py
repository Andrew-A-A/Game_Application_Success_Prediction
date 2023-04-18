import numpy as np
import pandas as pd
from numpy.ma import count
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def remove_primary_genre():
    df['Genres'] = list(df['Genres'].apply(lambda x: ', '.join(x.split(', ')[1:])))


df = pd.read_csv("games-regression-dataset.csv")

# Fill missing values in "In-app purchase" column with zero
df["In-app Purchases"] = df["In-app Purchases"].fillna(0)

# Drop unimportant columns
df = df.drop('URL', axis=1)
df = df.drop('ID', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('Subtitle', axis=1)
df = df.drop('Icon URL', axis=1)
df = df.drop('Description', axis=1)

# Calculate the mode of column 'Languages'
mode = df['Languages'].mode().iloc[0]

# Fill missing values in column 'Languages' with the mode
df['Languages'] = df['Languages'].fillna(mode)

# change datatypes from object
df = df.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
df['Age Rating'] = df['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
df['Age Rating'] = df['Age Rating'].astype(int)

remove_primary_genre()

unique = set(df['Genres'].str.replace('[^a-zA-Z]', '').str.lower().str.split().sum())
print(list(sorted(unique)))
print(count(list(sorted(unique))))

# Encode categorical columns
categorical_columns = ('Developer', 'Languages', 'Primary Genre')
feature_encoder(df, categorical_columns)

# calculate the sum of 'In-app Purchases'
df['In-app Purchases'] = df['In-app Purchases'].apply(lambda x: sum([float(num.strip(',')) for num in str(x).split()]))

# Extract features from date columns (Date => Day, Month, Year)
df['Original Release Date'] = pd.to_datetime(df['Original Release Date'], dayfirst=True)
df['Original Release Date_month'] = df['Original Release Date'].dt.month
df['Original Release Date_day'] = df['Original Release Date'].dt.day
df['Original Release Date_year'] = df['Original Release Date'].dt.year
df = df.drop('Original Release Date', axis=1)

df['Current Version Release Date'] = pd.to_datetime(df['Current Version Release Date'], dayfirst=True)
df['Current Version Release Date_month'] = df['Current Version Release Date'].dt.month
df['Current Version Release Date_day'] = df['Current Version Release Date'].dt.day
df['Current Version Release Date_year'] = df['Current Version Release Date'].dt.year
df = df.drop('Current Version Release Date', axis=1)

# Convert the string in column Genres into a list
df['Genres'] = df['Genres'].apply(lambda x: x.replace(' ', '').split(','))

# Split the values in column Genres into multiple rows
df = df.explode('Genres')

# Apply one-hot encoding to column Genres
df = pd.get_dummies(df, columns=['Genres'])

# Drop the duplicates
df = df.drop_duplicates()

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


Y = df['Average_User_Rating']
X = df.drop('Average_User_Rating', axis=1)

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=0)
print(df.dtypes)
count = 0
selected_features = []
for col in X_train.columns:
    x_current = X_train.loc[:, col]
    linear_model = LinearRegression()
    y_test = np.array(y_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    x_current = np.array(x_current).reshape(-1, 1)
    temp = X_test.loc[:, col]
    temp = np.array(temp).reshape(-1, 1)
    linear_model.fit(x_current, y_train)
    if metrics.mean_squared_error(y_train, linear_model.predict(x_current)) <= 0.5716:
        count += 1
        selected_features.append(col)
        print(f"Train mean sqError for{col} {metrics.mean_squared_error(y_train, linear_model.predict(x_current))} \n")
        print(f"Test mean sqError for {col} {metrics.mean_squared_error(y_test, linear_model.predict(temp))} \n")
print(f"counter = {count}")
df = df[selected_features]
df = df.join(Y)
print(selected_features)
cols = ["Price", "In-app Purchases", "User Rating Count", "Age Rating", "Size"]
X_num = X_train.loc[:, cols]

# Drop the duplicates
df = df.drop_duplicates()

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

# Feature Scaling to the whole data including all the X features and Y feature
column_names = df.columns.tolist()
standardization = preprocessing.StandardScaler()
scaled_data = standardization.fit_transform(df)
df = pd.DataFrame(scaled_data, columns=column_names)
# generate the profile report
profile = df.profile_report()

# save the report as a html file
profile.to_file(output_file='after_removingOutlier.html')

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
