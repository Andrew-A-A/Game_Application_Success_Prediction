import pandas as pd
from Preprocessing import *
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

new_columns = []
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
    # print(features)
    # Calculate the average TF-IDF value for each row
    max_tfidf = features.max(axis=1)
    max_tfidf = max_tfidf.todense().A1
    # Assign the average TF-IDF values to a new column in the data frame
    description_column = max_tfidf
    return description_column




def preprocess_test_data(x_test, y_test, unimportant_columns, global_vars, dev_encoder, lang_encoder,
                         primary_genre_encoder,
                         top_feature, x_train, unique_genres, standardization):
    # Drop unimportant data from the test data
    x_test = drop_columns(x_test, unimportant_columns)

    x_test['Description'] = feature_extraction(x_test['Description'])
    # Replace the list in 'In-app Purchases' column with the sum of the list in each entry
    x_test['In-app Purchases'] = calc_sum_of_list(x_test['In-app Purchases'])

    # Change data type to "datetime" in the 'Original Release Date' and 'Current Version Release Date' columns
    x_test['Original Release Date'] = pd.to_datetime(x_test['Original Release Date'], errors='coerce',
                                                     format='%d/%m/%Y')
    x_test['Current Version Release Date'] = pd.to_datetime(x_test['Current Version Release Date'], errors='coerce',
                                                            format='%d/%m/%Y')

    # Remove the '+' sign from the 'Age rating' column
    x_test['Age Rating'] = x_test['Age Rating'].str.replace('+', '', regex=False)

    # Convert the 'Age rating' column to an integer data type
    x_test['Age Rating'] = x_test['Age Rating'].astype(int)

    # x_test['Languages'] = fill_nulls_with_mode(x_train['Languages'])
    # print(x_train.dtypes)
    global_vars['Genres'] = global_vars['Primary Genre']

    x_test = x_test.apply(replace_genres_missing_vals, axis=1)
    # Fill missing values
    for col in x_test.columns:
        if col == 'In-app Purchases' or col == 'Price':
            x_test[col] = fill_nulls(x_test[col], 0)
        else:
            x_test[col].fillna(global_vars[col], inplace=True)

    # Extract feature (Difference in days) from 'Original Release Date' and 'Current Release Date'
    x_test['Difference in Days'] = (x_test['Current Version Release Date'] - x_test['Original Release Date']).dt.days

    # Drop both Original Release Data and Current Version Release Date
    x_test.drop(['Original Release Date', 'Current Version Release Date'], axis=1, inplace=True)

    test_data = x_test.join(y_test)
    test_data = remove_special_chars(test_data, 'Developer')
    y_test = test_data['Average User Rating']
    x_test = test_data.drop('Average User Rating', axis=1)

    x_test['Developer'] = dev_encoder.transform(x_test['Developer'])
    x_test['Languages'] = lang_encoder.transform(x_test['Languages'])
    x_test['Primary Genre'] = primary_genre_encoder.transform(x_test['Primary Genre'])

    # change datatypes from object
    x_test = x_test.convert_dtypes()

    # Remove the primary genre from the "Genres" feature
    x_test['Genres'] = remove_first_word(x_test['Genres'])

    # Change "Genres" values from string to list of strings
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
    y_test = x_test_data['Average User Rating']
    x_test = x_test_data.drop('Average User Rating', axis=1)
    return x_test, y_test


def cross_validation(model, x_train, y_train, n_splits):
    k_folds = KFold(n_splits)
    scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=k_folds)
    model_score = abs(scores.mean())
    return model_score
