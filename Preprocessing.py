from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from CustomLabelEncoder import *

"""
Global dictionary that will store either mean of feature (for numerical data) 
or mode (for categorical data) in order to fill nulls in testing data with this values """
# Key => column name, Value=>(mean/mode)
global_vars = {}

lbl = CustomLabelEncoder()
standardization = preprocessing.StandardScaler()

selected_features = []


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


# Apply cross validation
def cross_validation(model, x_train, y_train):
    k_folds = KFold(n_splits=5)
    scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=k_folds)
    model_score = abs(scores.mean())
    print("model 1 cross validation score is " + str(model_score))


def replace_genres_missing_vals(row):
    if not row['Primary Genre'] == '' and row['Genres'] == '':
        row['Genres'] = row['Primary Genre']
    elif not row['Genres'] == '' and row['Primary Genre'] == '':
        row['Primary Genre'] = row['Genres'].iloc[0]
    elif row['Primary Genre'] == '' and row['Genres'] == '':
        row['Genres'] = global_vars['Genres']
        row['Primary Genre'] = global_vars['Primary Genre']
    return row


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
