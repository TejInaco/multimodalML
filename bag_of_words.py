import nltk
from nltk.corpus import stopwords
import pandas as pd
import string
from collections import Counter

nltk.download('stopwords')

# load doc into memory
def load_doc(filename):
    data = pd.read_csv(filename, error_bad_lines=False)
    list_text = data['productDisplayName']
    return list_text

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = []
    for line in doc:
        if isinstance(line, str):
            words = line.split()
            for word in words:
                tokens.append(word)
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

if __name__ == "__main__":
    # load the document
    filename = 'data/styles.csv'
    text = load_doc(filename)
    tokens = clean_doc(text)
    # print(tokens)

    # define vocab
    vocab = Counter()
    vocab.update(tokens)
    print(len(vocab))
    print(vocab.most_common(50))

    #keep tokens with a min occurrence
    min_occurane = 2
    tokens = [k for k,c in vocab.items() if c >= min_occurane]
    print(len(tokens))

    # save tokens to file
    def save_list(lines, filename):
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()
 
    # save tokens to a vocabulary file
    save_list(tokens, 'data/vocab.txt')