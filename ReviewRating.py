import csv
import codecs
import platform
from sklearn.feature_extraction.text import TfidfVectorizer
from pylab import *
import scipy.spatial as sp
import scipy.sparse as sps
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
            next(file)
            reader = csv.reader(file)

            for row in reader:
                result.append(row[1])
                app_ids.append(row[0])
    return result, app_ids


"""
    Function to generate tfidf matrix for reviews

    :param fields: reviews data
    :return: return the reviews tfidf csr matrix
    :rtype: csr_matrix
"""


def generate_tfidf_reviews(data):
    review_tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english', norm='l2')
    review_tfidf_matrix = review_tf.fit_transform(data)
    review_feature_names = review_tf.get_feature_names()
    print(type(review_tfidf_matrix))
    print(review_tfidf_matrix)
    print(review_feature_names)
    return review_tfidf_matrix


"""
    Function to generate tfidf matrix for commit messages

    :param fields: commit messages
    :return: return the commit messages tfidf csr matrix
    :rtype: csr_matrix
"""


def generate_tfidf_commit_msgs(commits_text):
    commits_tf = TfidfVectorizer(analyzer='word', min_df=1, stop_words='english', norm='l2')
    commits_tfidf_matrix = commits_tf.fit_transform(commits_text)
    commits_feature_names = commits_tf.get_feature_names()
    print(type(commits_tfidf_matrix))
    print(commits_tfidf_matrix)
    print(commits_feature_names)
    return commits_tfidf_matrix


"""
    Generic function to plot the distance metrics graphs

    :param fields: distance metric ndarray
    :return: graphical representation
    :rtype: graph
"""


def plot_graph(cosine):
    plt.plot(cosine, color='blue')
    plt.xlabel("Reviews")
    plt.ylabel("Distance with commits message")
    plt.title("Review Commits Distance")
    plt.show()


def main():
    path = input("Please enter the reviews file path: ")
    path1 = input("Please enter the commits file path: ")

    df, app_ids = read_reviews_file(path) #traverse review file
    review_tfidf_matrix = generate_tfidf_reviews(df)

    print("*************************************************************************************************************")

    commits_text, commit_app_id = read_commits_file(path1) #traverse commit messages file
    commits_tfidf_matrix = generate_tfidf_commit_msgs(commits_text)

    #reshape commits matrix
    tf = sps.csr_matrix((commits_tfidf_matrix.data, commits_tfidf_matrix.indices, commits_tfidf_matrix.indptr),
                        shape=(80, 5186))

    #compute cosine matrix
    cosine = 1 - sp.distance.cdist(review_tfidf_matrix.todense(), tf.todense(), 'cosine')
    #plot_graph(cosine)

    #pearson correlation = negative (one value increase other decreases)
    correlation = 1- sp.distance.cdist(review_tfidf_matrix.todense(), tf.todense(), 'correlation')
    #print(correlation)

    #hamming distance gives proportion of vector elements which disagree, 1- done to get agreed elements
    hamming = 1- sp.distance.cdist(review_tfidf_matrix.todense(), tf.todense(), 'hamming')
    # print(hamming)
    # plot_graph(hamming)

    dice = 1- sp.distance.cdist(review_tfidf_matrix.todense(), tf.todense(), 'dice')
    #print(dice)
    #plot_graph(dice)


if __name__ == '__main__':
    print(platform.architecture())
    main()
