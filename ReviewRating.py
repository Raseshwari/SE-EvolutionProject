from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
import csv
import codecs
import platform
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.spatial as sp
np.set_printoptions(threshold=np.nan)




"""
    Function to read csv_reviews_file

    :param fields: local path where the dataset file is stored
    :return: return the user_story_ids and concatenated three columns of features
    :rtype: list, list
"""


def read_reviews_file(path):
    result = []
    app_ids = []
    with open(path, "r") as file:
        next(file)
        reader = csv.reader(file)

        for row in reader:
            result.append(row[2])
            app_ids.append(row[0])
    return result, app_ids


"""
    Function to read csv_reviews_file

    :param fields: local path where the dataset file is stored
    :return: return the user_story_ids and concatenated three columns of features
    :rtype: list, list
"""


def read_commits_file(path):
    result = []
    app_ids = []
    types_of_encoding = ["utf8", "cp1252"]
    for encoding_type in types_of_encoding:
        with codecs.open(path, encoding=encoding_type, errors='replace') as file:
        #with open(path, encoding = encoding_type, 'r') as file:
            next(file)
            reader = csv.reader(file)

            for row in reader:
                result.append(row[1])
                app_ids.append(row[0])
    return result, app_ids

"""
    Function lemmatizes the word tokens and returns a list of lemmatized token words

    :param fields: list of tokens to be lemmatized
    :return: list of lemmatized tokens
    :rtype: list
"""


def lemmation(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    lem_token_list = []
    for w in tokens:
        lem_token_list.append(wordnet_lemmatizer.lemmatize(w))
    return lem_token_list


"""
    Function for pos_tag to include only nouns, verbs, adverbs and adjectives

    :param fields: list of tokens from which part of speech tokens have to be retained
    :return: list of parts of speech tokens
    :rtype: list
"""


def pos_text(tokens):
    pos_tag_list = []
    pos_tokens = [word for word, pos in tokens if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'
                                                   or pos == "RB" or pos == "RBR" or pos == "RBS" or pos == "JJ"
                                                   or pos == "VB" or pos == "VBD" or pos == "VBG" or pos == "VBN" or pos == "VBP" or pos == "VBZ")]
    pos_tag_list.append(pos_tokens)
    return pos_tag_list


"""
    Function preprocesses text to lower case and removes standard and custom stop words

    :param fields: list of initial tokens 
    :return: list of preprocessed tokens
    :rtype: list
"""


def pre_process(details_list):
    custom_stopWords = ["hello", "thought", "asap", "licenses", "temp", "n't", "'s", ".", "'", "'ve", "!", "'m",
                        ":", "-", "e.g.", "house", "(", ")", ";", "default", "import", "void", "manager",
                        "asserts", "warnings", "params", "poorly", "review", "stars", "https", "\\n", "fixed"]
    stopset = set(stopwords.words('english'))
    tokens_list = []

    for i in range(len(details_list)):
        str = (details_list[i]).lower()
        tokens = word_tokenize(str)
        tokens = [w for w in tokens if not w in stopset]
        tokens = [w for w in tokens if not w in custom_stopWords]
        tokens = lemmation(tokens)
        # tokens = pos_text(tokens)
        tokens_list.append(pos_text(pos_tag(tokens)))
    return tokens_list



"""
    Function to traverse all the documents and compute a vocabulary corpus containing unique words from all the documents

    :param fields: list of preprocessed tokens 
    :return: set of vocabulary corpus containing unique tokens
    :rtype: set
"""


def compute_vocab_list(tokens_list):
    vocab_set = set()
    counter = 0
    for list in tokens_list:
        for sublist in list:
            for doc in sublist:
                counter += 1
                vocab_set.add(doc)
    return vocab_set


"""
    Function to compute frequency of term in a document

    :param fields: term and the vector(document) 
    :return: count of the times the term has occured in the document
    :rtype: int
"""


def freq(term, document):
    return document.count(term)


"""
    Function to compute term frequency of terms in each document

    :param fields: all the tokens in processed text
    :return: the term frequency for each document
    :rtype: list
"""


def compute_tf(processed_text):
    document = []
    counter = 0
    tf_doc = compute_vocab_list(processed_text)

    for i in processed_text:
        for doc in i:
            tf_vector = [freq(word, doc) for word in tf_doc]
            counter += 1
            document.append(tf_vector)
    return document


"""
    Function for computing l2 normalization

    :param fields: vector of each document
    :return: the term frequency for each document
    :rtype: list
"""


def l2_normalizer(vect):
    denominator = np.sum([element ** 2 for element in vect])
    if denominator == 0:
        return 0
    else:
        return [(element / math.sqrt(denominator)) for element in vect]


"""
    Function for computing normalized document matrix

    :param fields: document
    :return: computed list of documents for term frequency
    :rtype: list
"""


def compute_document_matrix(document):
    doc_term_matrix_l2_norm = []
    for vect in document:
        doc_term_matrix_l2_norm.append(l2_normalizer(vect))
    return doc_term_matrix_l2_norm


"""
    Function for counting number of docs having the word

    :param fields: word to be counted and document list
    :return: list of documents which contain the word
    :rtype: list
"""


def number_of_docs_with_word(word, document_list):
    document_count = 0
    for doc in document_list:
        if freq(word, doc) > 0:
            document_count += 1
    return document_count


"""
    Function to compute the idf value for each word in the document list

    :param fields: word to be counted and document list
    :return: list of documents which contain the word
    :rtype: float
"""


def compute_idf_value(word, document_list):
    number_of_documents = len(document_list)
    df = number_of_docs_with_word(word, document_list)
    return np.log(number_of_documents / 1 + df)


"""
    Function to compute idf matrix

    :param fields: idf_vector
    :return: list of documents which contain the word
    :rtype: ndarray
"""


def compute_idf_matrix(idf_vector):
    idf_matrix = np.zeros((len(idf_vector), len(idf_vector)))  # Fill in zeroes where there is no value
    np.fill_diagonal(idf_matrix, idf_vector)  # Fill the values diagonally
    return idf_matrix


"""
    Function to calculate the dot product of normalized tf and idf matrix

    :param fields: term frequency matrix and inverse document frequency matrix
    :return: normalized tf_idf matrix
    :rtype: matrix
"""


def compute_tf_idf(normalized_doc_matrix, idf_matrix):
    doc_term_matrix_tfidf = []
    for tf_vector in normalized_doc_matrix:
        doc_term_matrix_tfidf.append(np.dot(tf_vector, idf_matrix))

    doc_term_matrix_tfidf_l2 = []
    for tf_vector in doc_term_matrix_tfidf:
        doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))
    return np.matrix(doc_term_matrix_tfidf_l2)


def main():
    path = input("Please enter the reviews file path: ")
    path1 = input("Please enter the commits file path: ")

    df, app_ids = read_reviews_file(path)
    #print(df)

    # review_processed_text = pre_process(df)
    #
    # review_vocabulary = compute_vocab_list(review_processed_text)
    # review_document = compute_tf(review_processed_text)  # list of size 1000 with 1/0 comma-seperated for each word
    #
    # review_normalized_doc_matrix = compute_document_matrix(review_document)
    #
    # # For every word in the vocabulary count documents word is present in and compute word's idf value
    # review_idf_vector = [compute_idf_value(word, review_processed_text) for word in review_vocabulary]
    # review_idf_matrix = compute_idf_matrix(review_idf_vector)
    #
    # review_doc_term_matrix_tfidf_l2 = compute_tf_idf(review_normalized_doc_matrix, review_idf_matrix)
    # print(type(review_doc_term_matrix_tfidf_l2))

    review_tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english', norm='l2')
    review_tfidf_matrix = review_tf.fit_transform(df)
    review_feature_names = review_tf.get_feature_names()
    print(type(review_tfidf_matrix))
    print(review_tfidf_matrix)
    print(review_feature_names)

    print("*************************************************************************************************************")
    commits_text, commit_app_id = read_commits_file(path1)
    #print(commits_text)
    # commits_processed_text = pre_process(commits_text)
    #
    # commits_vocabulary = compute_vocab_list(commits_processed_text)

    commits_tf = TfidfVectorizer(analyzer='word', min_df=1, stop_words='english', norm='l2')
    commits_tfidf_matrix = commits_tf.fit_transform(commits_text)
    commits_feature_names = commits_tf.get_feature_names()
    print(type(commits_tfidf_matrix))
    print(commits_tfidf_matrix)
    print(commits_feature_names)

    # commits_document = compute_tf(commits_processed_text)  # list of size 1000 with 1/0 comma-seperated for each word
    #
    # commits_normalized_doc_matrix = compute_document_matrix(commits_document)

    # For every word in the vocabulary count documents word is present in and compute word's idf value
    # commits_idf_vector = [compute_idf_value(word, commits_processed_text) for word in commits_vocabulary]
    # commits_idf_matrix = compute_idf_matrix(commits_idf_vector)

    #commits_doc_term_matrix_tfidf_l2 = compute_tf_idf(commits_normalized_doc_matrix, commits_idf_matrix)
    #print(df)

    print("***********************************************************************************************************")
    # print(np.array(commits_tfidf_matrix.todense()).zeros(np.array(review_tfidf_matrix.todense().shape)), np.array(review_tfidf_matrix.todense().shape))
    # c_tfidf = np.array(commits_tfidf_matrix).reshape(120, 5186).shape
    # print(c_tfidf.shape)


    # cosine = 1 - sp.distance.cdist(commits_tfidf_matrix.todense(), review_tfidf_matrix.todense(), 'cosine')
    # print(cosine)

    #nrow = review_tfidf_matrix.shape[0]
    # for i in range(nrow):
    #     column = []
    #     for j in range(commits_tfidf_matrix.shape[0]):
    #         column.append(1-sp.distance.cdist(review_tfidf_matrix[i],commits_tfidf_matrix[j], 'cosine'))

    print(np.array(review_tfidf_matrix.todense()))
    print(np.array(commits_tfidf_matrix.todense()))



if __name__ == '__main__':
    print(platform.architecture())
    main()