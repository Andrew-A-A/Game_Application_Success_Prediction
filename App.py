import Test
from Preprocessing import *
from sklearn.model_selection import train_test_split

# Load the csv file
df = pd.read_csv("games-regression-dataset.csv")

# Split data frame to X and Y
Y = df['Average_User_Rating']
X = df.drop('Average_User_Rating', axis=1)

# Split the X and the Y to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=0)

# Drop unimportant columns
unimportant_columns = ['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'Description']
x_train = drop_columns(x_train, unimportant_columns)

# Fill missing values in "In-app purchases" column with zero
x_train['In-app Purchases'] = fill_nulls(x_train['In-app Purchases'], 0)

# Fill missing values in column 'Languages' with the mode
x_train['Languages'] = fill_nulls_with_mode(x_train['Languages'])

# change datatypes from object
x_train = x_train.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
x_train['Age Rating'] = x_train['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
x_train['Age Rating'] = x_train['Age Rating'].astype(int)

# Remove the primary genre from the "Genres" feature
x_train['Genres'] = remove_first_word(x_train['Genres'])

# Encode categorical columns
categorical_columns = ('Developer', 'Languages', 'Primary Genre')
feature_encoder_fit(x_train, categorical_columns)

# calculate the sum of 'In-app Purchases'
x_train['In-app Purchases'] = calc_sum_of_list(x_train['In-app Purchases'])

# Split "Original release date" and "Current version date" into three columns
# (Original release date_day, Original release date_month, Original release date_year)
x_train = explode_date(x_train, 'Original Release Date')

# (Current Version Release Date_day, Current Version Release Date_month, Current Version Release Date_year)
x_train = explode_date(x_train, 'Current Version Release Date')

# Replace outliers
x_train = outlier_iqr_replace(x_train)


# Apply hot one encoding for the "Genres" column
train_data = x_train.join(y_train)
train_data, Test.new_columns = hot_one_encode(train_data, 'Genres')
y_train = train_data['Average_User_Rating']
x_train = train_data.drop('Average_User_Rating', axis=1)

calc_modes_of_features(x_train)
x_test, y_test = Test.preprocess_test_data(x_test, y_test)
x_test = x_test.loc[:, x_train.columns]

# Feature selection
x_train = wrapper_feature_selection(x_train, y_train, x_test, y_test)


x_test = x_test[selected_features]


# Apply feature standardization
test_data = x_test.join(y_test)
train_data = x_train.join(y_train)
train_data = feature_standardization_fit(train_data)
test_data = Test.feature_standardization_apply(test_data)
