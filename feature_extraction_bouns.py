# import nltk
# import pandas as pd
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
#
# lemmatizer = WordNetLemmatizer()
# vectorizer = TfidfVectorizer()
#
# def preprocess_text(text):
#     words = word_tokenize(text.lower())
#     words = [lemmatizer.lemmatize(w) for w in words]
#     s_words = set(words)
#     return s_words
#
# def feature_extraction(col):
#     returned_list = []
#     for description in col:
#         words = preprocess_text(description)
#         meaningful_words = []
#         for word in words:
#             if len(word) < 3:
#                 continue
#             synsets = wordnet.synsets(word)
#             if synsets:
#                 meaningful_words.append(word)
#
#         pos_tags = nltk.pos_tag(meaningful_words)
#         nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
#         nouns = ' '.join(nouns)
#         returned_list.append(nouns)
#     returned_list = pd.DataFrame({'New':returned_list})
#     print(returned_list)
#     features = vectorizer.fit_transform(returned_list)
#     print(features)
#     return features
# #
# df = pd.read_csv("E:\MehraelAshraf\Kolia\\6th semester\Pattern\Project Materials\Datasets\Milestone 1\games-regression-dataset.csv")
# nouns = feature_extraction(df['Description'])
#
#
# # print(features)
# # print(nouns)
# #
# # # Embedding Encoding
# # tokenizer = Tokenizer()  # Initialize tokenizer
# # tokenizer.fit_on_texts(nouns['New'])  # Fit tokenizer on input texts
# # word_index = tokenizer.word_index  # Get word index
# # sequences = tokenizer.texts_to_sequences(nouns['New'])  # Convert input texts to sequences
# # max_len = max([len(seq) for seq in sequences])  # Get maximum sequence length
# # embeddings = pad_sequences(sequences, maxlen=max_len, padding='post')  # Pad sequences to the same length
# # print("Embedding Encoding:")
# # print("Word Index: ", word_index)
# # # print("Sequences: ", sequences)
# # print("Embeddings: \n", embeddings)
# # print()
# #
# #
# # # One-Hot Encoding
# # onehot_encoder = OneHotEncoder()  # Initialize one-hot encoder
# # onehot_encoded = onehot_encoder.fit_transform(sequences).toarray()  # Convert sequences to one-hot encoded vectors
# # print("One-Hot Encoding:")
# # print("One-Hot Encoded Vectors: \n", onehot_encoded)