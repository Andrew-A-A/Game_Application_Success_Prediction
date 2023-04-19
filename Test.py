from Preprocessing import *

new_columns = []


def feature_encoder_apply(test, columns):
    for c in columns:
        fill_nulls(test, global_vars[c])
        test[c] = lbl.transform(list(test[c].values))
    return test


def hot_one_encode(df, column, new_columns=None):
    y = df['Average_User_Rating']
    # Convert the string in column Genres into a list
    df[column] = df[column].apply(lambda x: x.replace(' ', '').split(','))
    # Split the values in column Genres into multiple rows
    df = df.explode(column)

    # Apply one-hot encoding to column Genres
    df_encoded = pd.get_dummies(df, columns=[column])

    # Add any missing columns
    if new_columns is not None:
        missing_columns = new_columns.difference(df_encoded.columns)
        for col in missing_columns:
            df_encoded[col] = 0
    # Drop the duplicates
    df_encoded = df_encoded.drop_duplicates()

    return df_encoded


def feature_standardization_apply(test):
    column_names = test.columns.tolist()
    scaled_data = standardization.transform(test)
    test = pd.DataFrame(scaled_data, columns=column_names)
    return test


def preprocess_test_data(x, y):
    # Drop unimportant columns
    unimportant_columns = ['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'Description']
    x = drop_columns(x, unimportant_columns)
    print(x['Genres'].isnull().sum())
    # Fill null values
    for col in x.columns:
        if col == 'In-app Purchases':
            x[col] = fill_nulls(x[col], 0)
        else:
            if global_vars.get(col) is not None:
                x[col] = fill_nulls(x[col], global_vars[col])

    # change datatypes from object
    x = x.convert_dtypes()

    # Remove the '+' sign from the 'Age rating' column
    x['Age Rating'] = x['Age Rating'].str.replace('+', '', regex=False)

    # Convert the 'Age rating' column to an integer data type
    x['Age Rating'] = x['Age Rating'].astype(int)

    # Remove the primary genre from the "Genres" feature
    x['Genres'] = remove_first_word(x['Genres'])

    # Encode categorical columns
    categorical_columns = ('Developer', 'Languages', 'Primary Genre')
    feature_encoder_apply(x, categorical_columns)

    # calculate the sum of 'In-app Purchases'
    x['In-app Purchases'] = calc_sum_of_list(x['In-app Purchases'])

    # Split "Original release date" and "Current version date" into three columns
    # (Original release date_day, Original release date_month, Original release date_year)
    x = explode_date(x, 'Original Release Date')

    # (Current Version Release Date_day, Current Version Release Date_month, Current Version Release Date_year)
    x = explode_date(x, 'Current Version Release Date')

    # Apply hot one encoding for the "Genres" column
    test_data = x.join(y)
    test_data = hot_one_encode(test_data, 'Genres', new_columns)
    y = test_data['Average_User_Rating']
    x = test_data.drop('Average_User_Rating', axis=1)

    return x, y
