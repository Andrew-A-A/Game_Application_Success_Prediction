import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def feature_encoder(x, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x


df = pd.read_csv("games-regression-dataset.csv")

# Fill missing values in "In-app purchase" column with zero
df["In-app Purchases"] = df["In-app Purchases"].fillna(0)

# Drop subtitle column
df = df.drop('Subtitle', axis=1)

# Calculate the mode of column 'Languages'
mean = df['Languages'].mode().iloc[0]

# Fill missing values in column 'Languages' with the mode
df['Languages'] = df['Languages'].fillna(mean)

# Drop the duplicates
df = df.drop_duplicates()
df = df.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
df['Age Rating'] = df['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
df['Age Rating'] = df['Age Rating'].astype(int)

# Encode categorical columns
categorical_columns = ('URL', 'Name', 'Icon URL', 'Description', 'Developer',
                       'Languages', 'Primary Genre', 'Genres')
feature_encoder(df, categorical_columns)

# Extract features from date columns (Date => Day, Month, Year)
df['Original Release Date'] = pd.to_datetime(df['Original Release Date'], dayfirst=True)
df['Current Version Release Date'] = pd.to_datetime(df['Current Version Release Date'], dayfirst=True).dt.date


print(df.dtypes)


def feature_scaling(x, a, b):
    x = np.array(x)
    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i] - min(x[:, i])) / (max(x[:, i]) - min(x[:, i]))) * (b - a) + a
    return normalized_x
