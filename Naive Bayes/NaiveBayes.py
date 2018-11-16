#! user/bin/pycharmPoject

import numpy as np


# segment text
def text_segment(large_string):

    """
    Desc: Receive a large string and segment it to a string list
    :param large_string:
    :return: String list and turn all string into lower case
    """

    import re
    clean_string = re.compile(r'[^a-zA-z]|\d')
    list_string = clean_string.split(large_string)
    list_string_lower = [string.lower() for string in list_string if len(list_string) > 0]
    return list_string_lower


def loadData(filename):
    """
    Desc: Load date from scv file
    :param filename:
    :return: A list marked whether each mail is spam using 1 or 0. And a list contains content of these mails
    """
    category = []
    textWords= []
    import csv
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['v1'])
            if row['v1'] == 'ham':
                category.append(0)
            elif row['v1'] == 'spam':
                category.append(1)
                words = text_segment(row['v2'])
                textWords.append(words)
    return category, textWords


def creat_vocable_list(textWords):
    """
    Desc: obtain the set of all words
    :param dataset:
    :return: vocable_list(no repetition)
    """
    vocable_set = set([])
    for document in textWords:
        vocable_set = vocable_set | set(document)
        vocableList = vocable_set
    return vocableList


def setOfwordsToVoc(vocableList, inputList):
    """
    Desc: find out whether the words in inputList appear in vocable list
    :param vocableList:
    :param inputList:
    :return: the resList, if one word is appeared the corresponding index of the word in resList is 1, else is 0
    """
    resList = [0] * vocableList
    for word in inputList:
        if word in vocableList:
            resList[vocableList.index(word)] = +1
    return resList


def setOfwordslistToVoc(vocableList, wordsList):
    """
    Desc:
    :param vocableList:
    :param wordsList:
    :return:
    """
    setOfwordList = []
    for i in range(len(wordsList)):
        setOfword = setOfwordsToVoc(vocableList, wordsList[i])
        setOfwordList.append(setOfword)
        return setOfwordList


def trainNB(trainMatrix, trainCategory):
    """
    Desc:
    :param trainMatrix:
    :param trainCategory:
    :return:
    """
    numOfmails = len(trainMatrix)
    numOfwords = len(trainMatrix[0])
    #P(s)
    Pspam = sum(trainCategory)/float(numOfmails)
    # initial each word as 1 in mails
    WordsFecOfSpam = np.ones(numOfwords)
    WordsFecOfHam = np.ones(numOfwords)
    numWordsInSpam = 2.0
    numWordsInHam = 2.0
    for i in range(0,numOfmails):
        if trainCategory[i] == 1:
            WordsFecOfSpam += trainMatrix[i]
            numWordsInSpam += sum(trainMatrix[i])
        else:
            WordsFecOfHam += trainMatrix[i]
            numWordsInHam += sum(trainMatrix[i])
    PwordsOfSpam = np.log(WordsFecOfSpam/numWordsInSpam)
    PwordsOfHam = np.lpg(WordsFecOfHam/numWordsInHam)
    return Pspam, PwordsOfSpam, PwordsOfHam


def classify():
    return
