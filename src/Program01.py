"""
Content of this file
1. SPARQL-query (Output: Dataframe of all keywords and their identifiers)
2. Converts into a corpus (Output: List of strings, string includes all keywords for a given identifier)
2.1 Save to csv and read from csv
3. Tokenize and Vectorize
3.1 Find information about the keywords (which one is used most often, which ones are only used once)
4. Build the CBOW model
4.1 Reduce dimensionality (Output: List of two-dimensional tuples (?) with coordinates for every word in the vocabulary)
5. Visualize the model
6. Find good epsilon
7. DBSCAN
Then look into clusters and see if they make sense.
Then try to implement a clustering of the texts based on the vectors of the keywords

Add-Ons:
- Filter all keywords that are only being used once (and count them, I want to know how many there are)
    Dropping rare words:
    https://stackoverflow.com/questions/61460683/word2vec-using-document-body-or-keywords-as-training-corpus
- Why are there keywords like _gemeinden, _arbeitnehmer, _urlaub, _kreise, _städte, _ehrenamt? (ID ca. 28884)
"""

import urllib.parse
import requests
import pandas as pd
from io import StringIO
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from keras.preprocessing import text
import keras.utils as utils
from keras.preprocessing import sequence

"""1. SPARQL-Query"""

SPARQL_ENDPOINT = "https://www.govdata.de/sparql"
SPARQL_PREFIXES = """PREFIX dct: <http://purl.org/dc/terms/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>"""


def sparql_query_id_keywords() -> pd.DataFrame:
    # Builds a SPARQL query that gives out all keywords for a dataset with a specific identifier
    query = SPARQL_PREFIXES
    query += """SELECT ?title ?identifier ?keyword WHERE {
    ?dataset a dcat:Dataset .
    ?dataset dct:title ?title .
    ?dataset dct:identifier ?identifier .
    ?dataset dcat:keyword ?keyword .
    }"""
    url = SPARQL_ENDPOINT + "?query=" + urllib.parse.quote(query)
    r = requests.get(url, headers={"Accept": "text/csv"})
    data = StringIO(r.text)
    df = pd.read_csv(data)
    return df


# keywords = sparql_query_id_keywords()
# print('Download beendet')
# print(keywords)

"""
# Save to csv
keywords.to_csv('Program01_keywords.csv', encoding="utf-8")
# Retrieve from csv
#keywords = pd.read_csv('Program01_keywords.csv', encoding="utf-8", index_col=0)
#print('Dataframe eingelesen')
#print(keywords)
"""


"""2. Converts into a corpus"""


def convert_df_to_lists(df:pd.DataFrame) -> tuple[list, list, list]:
    identifiers = []  # List for all the identifiers
    titles = []  # List for all the titles
    corpus = []  # List for all the keywords
    for index, row in df.iterrows():
        if df["identifier"][index] in identifiers:
            continue
        identifiers.append(df["identifier"][index])
        titles.append(df["title"][index])
        corpus_entry = ""
        for i in df.index[index:]:
            if df["identifier"][i] != identifiers[-1]:
                break
            entry = re.sub(
                r"\s", "-", df["keyword"][i]
            )  # Changes whitespace characters to underlines, so as to avoid further issues with the tokenization
            corpus_entry = corpus_entry + entry + " "
        corpus.append(corpus_entry)
    return identifiers, corpus, titles


# identifiers, corpus, titles = convert_df_to_lists(keywords[:1000])
# print('Listen erstellt')


"""2.1 Save to csv and read from csv to list"""


# Save to csv
def save_csv(filename:str, lst, keyword:str) -> None:
    with open(filename, "w", encoding="utf-8", newline="") as f:
        write = csv.writer(f)
        if keyword == "mode1":
            for entry in lst:
                write.writerow([entry])
        if keyword == "mode2":
            for entry in lst:
                write.writerow(entry)
    print("CSV gespeichert")
    return


# save_csv('Program01_titles.csv', titles, 'mode1')
# save_csv('Program01_corpus.csv', corpus, 'mode1')
# save_csv('Program01_identifiers.csv', identifiers, 'mode1')


# Read to list from csv
def read_to_list(filename:str, keyword:str) -> list:
    lst = []
    with open(filename, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if keyword == "mode1":
            for line in reader:
                lst.append(line[0])
        if keyword == "mode2":
            for line in reader:
                lst.append(line)
    return lst


identifiers = read_to_list("Program01_identifiers.csv", "mode1")
corpus = read_to_list("Program01_corpus.csv", "mode1")
titles = read_to_list("Program01_titles.csv", "mode1")

# print('Anzahl der Datensätze: ', len(identifiers))

"""3. Tokenize and Vectorize"""

tokenizer = text.Tokenizer(filters="")
tokenizer.fit_on_texts(corpus)
word2id = tokenizer.word_index
word2id["PAD"] = 0
id2word = {v: k for k, v in word2id.items()}
sequences = tokenizer.texts_to_sequences(corpus)
# print(sequences)

# Print information about the dataset with the most keywords (hint: it has 720 keywords)
# print(max(corpus, key=len))
# print(max(sequences, key=len))
# print(len(max(sequences, key=len)))

"""
# Save the word index to csv
with open('Program01_word_index.csv', 'w', encoding="utf-8", newline='') as f:
    write = csv.writer(f)
    for key, value in tokenizer.word_index.items():
       write.writerow([key, value])
# Read to dict from csv
with open('Program01_word_index.csv', encoding="utf-8", newline='') as f:
    reader = csv.reader(f)
    word_index = dict(reader)
"""

"""3.1 Find information about the keywords """

# I have:
# identifiers = List of every identifier of every dataset
# titles = List of every title of every dataset
# corpus = List of lists of strings with the keywords of every dataset
# sequences = List of lists of integers corresponding to the kewords of every dataset
# tokenizer.word_index = Dict with every unique keyword and their ID
# I want:
# keyword_counts = Dict with every unique keywords and their total counts


def get_keyword_counts(sequences:list) -> dict:
    keyword_counts = {}
    for dataset in sequences:
        for id in dataset:
            if id in keyword_counts:
                keyword_counts[id] = keyword_counts[id] + 1
            else:
                keyword_counts[id] = 1
    return keyword_counts


# keyword_counts = get_keyword_counts(sequences)

"""
# Save to csv
with open('Program01_keyword_counts.csv', 'w', encoding="utf-8", newline='') as f:
    write = csv.writer(f)
    for key, value in keyword_counts.items():
       write.writerow([key, value])
# Read to dict from csv
with open('Program01_keyword_counts.csv', encoding="utf-8", newline='') as f:
    reader = csv.reader(f)
    keyword_counts = dict(reader)
#print(keyword_counts)
"""

# print('Most used keyword: ', max(keyword_counts, key=keyword_counts.get),
#       ', used ', max(keyword_counts.values()), ' times')
# print(list(keyword_counts.values()).count(1), ' out of ', len(keyword_counts) ,' keywords are being used only once')
# print('Keyword 1: ', tokenizer.word_index[1], ', used ', keyword_counts[1], ' times')


"""4. Build the CBOW model"""

# In this case with windows size 2
# But I want no window size!


def generate_context_word_pairs(corpus:list, window_size:int, vocab_size:int):
    context_length = window_size * 2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []
            start = index - window_size
            end = index + window_size + 1

            context_words.append(
                [
                    words[i]
                    for i in range(start, end)
                    if 0 <= i < sentence_length and i != index
                ]
            )
            label_word.append(word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = utils.to_categorical(label_word, vocab_size)
            yield (x, y)


def build_cbow_model(vocab_size:int, embedding_size:int, window_size:int) -> Sequential:
    # Define the CBOW model
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            input_length=2 * window_size,
        )
    )
    model.add(
        Lambda(lambda x: tf.reduce_mean(x, axis=1), output_shape=(embedding_size,))
    )
    model.add(Dense(units=vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    # model.save_weights('cbow_weights.h5')
    # Load the pre-trained weights
    # model.load_weights('cbow_weights.h5')
    print("Neuronales Netz gebaut")
    return model


# Define the parameters
vocab_size = len(word2id)
embedding_size = 2
window_size = 3
model = build_cbow_model(vocab_size, embedding_size, window_size)
# model.summary()

# Train the Neural Network
for epoch in range(1, 2):
    loss = 0.0
    i = 0
    for x, y in generate_context_word_pairs(
        corpus=sequences, window_size=window_size, vocab_size=vocab_size
    ):
        i += 1
        loss += model.train_on_batch(x, y)
        if i % 100000 == 0:
            print("Processed {} (context, word) pairs".format(i))
        print(i)

    print("Epoch:", epoch, "\tLoss:", loss)
    print()

weights = model.get_weights()[0]
weights = weights[1:]
df = pd.DataFrame(weights, index=list(id2word.values())[1:])
df.to_csv("Program01_distance-matrix_emb2_ep1.csv", encoding="utf-8")
model.save_weights("cbow_weights.h5")

"""4.1 Reduce dimensionality"""

# Get the word embeddings
# embeddings = model.get_weights()[0]


# Perform PCA to reduce the dimensionality
# of the embeddings
# pca = PCA(n_components=2)
# reduced_embeddings = pca.fit_transform(embeddings)

# save_csv('Program01_embeddings.csv', embeddings, 'mode2')
# save_csv('Program01_reduced_embeddings.csv', reduced_embeddings, 'mode2')

# embeddings = np.array(read_to_list('Program01_embeddings.csv', 'mode2'))
# reduced_embeddings = np.array(read_to_list('Program01_reduced_embeddings.csv', 'mode2'))

# Create list with keywords that only keeps keywords that are being used at least 5 times
# reduced_embeddings_shortened = reduced_embeddings[:13163]

"""5. Visualize the model"""

# Visualize the embeddings

# Visualization without words
# plt.scatter(reduced_embeddings_shortened[:, 0], reduced_embeddings_shortened[:, 1], s=1, alpha=0.5)
# plt.show()

"""
# Visualization with words
plt.figure(figsize=(5, 5))
for i, word in enumerate(word_index.keys()):
    x, y = reduced_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(word, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points',
                 ha='right', va='bottom')
    print(i)
plt.show()
"""

"""6. Find good epsilon

#word_index = tokenizer.word_index       # Holds the vocabulary in a dict. Keys = words, items = integer
#print(list(word_index.keys())[0:10])
#print(embeddings[0:10])                 # Holds the vectorized vocabulary in a nested list.
#                                          Each vector is a list of 10 floats
#print(reduced_embeddings[0:10])         # Same as above, but with two dimensions instead of 10

nn = NearestNeighbors(n_neighbors=4)
nbrs = nn.fit(reduced_embeddings_shortened)
distances, indices = nbrs.kneighbors(reduced_embeddings_shortened)

distances = np.sort(distances, axis=0)
distances = distances[:,1]

#plt.figure(figsize=(6,3))
plt.plot(distances)
#plt.axhline(y=0.24, color='r', linestyle='--', alpha=0.4) # elbow line
plt.show()

# epsilon -> 0.00150?
# epsilon evtl auch 0.00210
"""

"""7. DBSCAN


dbs = DBSCAN(eps=0.0019, min_samples=5)
dbs.fit(reduced_embeddings_shortened)

labels = list(dbs.labels_)

# Give some information about the results of the DBSCAN algorithm
print('All data points: ', len(reduced_embeddings))
print('All noisy data: ', labels.count(-1))
print('All clusters: ', np.unique(labels))

for i in range(5):
    print('Words of cluster', i+1, ': ')
    for datapoint in range(len(reduced_embeddings_shortened)):
        if labels[datapoint] == i+1:
            print(list(tokenizer.word_index)[datapoint], ',')
"""
