import nltk
from nltk.corpus import stopwords
import pandas as pd
import string
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
import random
from numpy import array
from pandas import DataFrame
from matplotlib import pyplot
from bag_of_words import clean_doc

nltk.download('stopwords')

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)

    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def evaluate_mode(X_train, y_train, X_test, y_test):
    scores = list()
    n_repeats = 2
    n_words = X_test.shape[1]
    for i in range(n_repeats):
        model = get_model(n_words)

        # fit network
        model.fit(X_train, y_train, epochs=5, verbose=1)

        # evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        scores.append(acc)
        print('%d accuracy: %s' % ((i+1), acc))

    return scores

def get_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(4, activation='sigmoid'))

    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def get_data(data):

    # load the vocabulary
    vocab_filename = 'data/vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    sentences = data['productDisplayName'].values.tolist()
    usage = pd.get_dummies(data['season'])
    usage = usage.values.tolist()

    # create the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    #construct train and test data
    split_num = int(len(sentences) * 0.7)
    train_data = sentences[:split_num]
    test_data = sentences[split_num:]
    y_train = array(usage[:split_num])
    y_test = array(usage[split_num:])
    X_train = tokenizer.texts_to_matrix(train_data, mode=mode)
    X_test = tokenizer.texts_to_matrix(test_data, mode=mode)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    vocab_filename = 'data/vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    data = pd.read_csv('data/styles.csv', error_bad_lines=False)

    sentences = data['productDisplayName'].values.tolist()
    usage = pd.get_dummies(data['season'])
    usage = usage.values.tolist()
    index_to_remove = []

    # remove bad data
    for i in range(0, len(sentences)):
        if not isinstance(sentences[i], str):
            index_to_remove.append(i)
    
    aux = 0
    for i in index_to_remove:
        sentences.pop(i - aux)
        usage.pop(i - aux)
        aux += 1

    # create the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    
    #TODO: Split into two in a even way

    #construct train and test data
    split_num = int(len(sentences) * 0.7)
    train_data = sentences[:split_num]
    test_data = sentences[split_num:]
    y_train = array(usage[:split_num])
    y_test = array(usage[split_num:])

    # making for every possible mode and see how it behaves
    results = DataFrame()   
    modes = ['binary', 'count', 'tfidf', 'freq']
    for mode in modes:
        X_train = tokenizer.texts_to_matrix(train_data, mode=mode)
        print(X_train.shape)
        print(y_train.shape)

        X_test = tokenizer.texts_to_matrix(test_data, mode=mode)
        print(X_test.shape)
        print(y_test.shape)

        results[mode] = evaluate_mode(X_train, y_train, X_test, y_test)

    # summarize results
    print(results.describe())

    # plot results
    results.boxplot()
    pyplot.show()