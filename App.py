from sklearn.model_selection import train_test_split
from Preprocessing import *
# Load the csv file
df = pd.read_csv("games-regression-dataset.csv")

# Drop unimportant columns
unimportant_columns = ['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'Description']
df = drop_columns(df, unimportant_columns)

# Fill missing values in "In-app purchases" column with zero
df['In-app Purchases'] = fill_nulls(df['In-app Purchases'], 0)

# Fill missing values in column 'Languages' with the mode
df['Languages'] = fill_nulls_with_mode(df['Languages'])

# change datatypes from object
df = df.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
df['Age Rating'] = df['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
df['Age Rating'] = df['Age Rating'].astype(int)

# Remove the primary genre from the "Genres" feature
df['Genres'] = remove_first_word(df['Genres'])

# Encode categorical columns
categorical_columns = ('Developer', 'Languages', 'Primary Genre')
feature_encoder(df, categorical_columns)

# calculate the sum of 'In-app Purchases'
df['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])

# Split "Original release date" and "Current version date" into three columns
# (Original release date_day, Original release date_month, Original release date_year)
df = explode_date(df, 'Original Release Date')

# (Current Version Release Date_day, Current Version Release Date_month, Current Version Release Date_year)
df = explode_date(df, 'Current Version Release Date')

# Apply hot one encoding for the "Genres" column
df = hot_one_encode(df, 'Genres')

# Split data frame to X and Y
Y = df['Average_User_Rating']
X = df.drop('Average_User_Rating', axis=1)

# Split the X and the Y to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=0)

# Feature selection
df = wrapper_feature_selection(df, x_train, y_train, x_test, y_test)

# Drop the duplicates
df = df.drop_duplicates()

# Replace outliers
df = outlier_iqr_replace(df)

# Apply feature scaling
df = feature_scale(df)
