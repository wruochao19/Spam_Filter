import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# use MultinomialNB algorithm
from sklearn.naive_bayes import MultinomialNB

# import method for split train/test data set
from sklearn.model_selection import train_test_split

# import method to calculate metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


def spam_filter():
    # prepare data
    data = pd.read_csv("../spam.csv", header=0, encoding="ISO-8859-1")
    data.head()
    #
    tfidf_vect = TfidfVectorizer()
    dtm = tfidf_vect.fit_transform(data["v2"])
    print("size of tfidf matrix:", dtm.shape)
    print("total number of words:", len(tfidf_vect.vocabulary_))
    voc_lookup = {tfidf_vect.vocabulary_[word]: word for word in tfidf_vect.vocabulary_}

    print("\nOriginal text: \n" + data["v2"][0])
    print("\ntfidf weights: \n")

    # covert the sparse matrix row to a dense array
    doc0 = dtm[0].toarray()[0]
    print(doc0.shape)

    # get index of top 20 words
    # top_words = (doc0.argsort())[::-1][0:20]
    # [(voc_lookup[i], doc0[i]) for i in top_words]

    # # split dataset for 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(dtm, data["v1"], test_size=0.3, random_state=0)
    clf = MultinomialNB().fit(X_train, y_train)
    #
    # predict the news group for the test dataset
    predicted = clf.predict(X_test)
    #
    labels = ['ham', 'spam']
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, predicted, labels=labels)
    print("labels: ", labels)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f-score: ", fscore)
    print("support: ", support)
    print(classification_report(y_test, predicted, target_names=labels))
    return tfidf_vect, clf

def predect_new(tfidf_vect, clf, text):
    labels = ['ham', 'spam']
    new_text_tfidf = tfidf_vect.transform(text)
    print(new_text_tfidf)
    print(new_text_tfidf.shape)
    predicted_p = clf.predict_proba(new_text_tfidf)
    predicted = clf.predict(new_text_tfidf)
    for idx, doc in enumerate(text):
        print('\n', doc)
        for j, label in enumerate(labels):
            print('% s: %.3f' % (labels[j], predicted_p[idx][j]))
        print('%r => %s' % (doc, predicted[idx]))

if __name__ == '__main__':
    """
        generate tfidf matrix and a NB model
    """
    tfidf_vect, clf = spam_filter()

    """
        input : list of email text
    """
    text = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", "U dun say so early hor... U c already then say..."]
    """
        output: 
    """
    predect_new(tfidf_vect, clf, text)
