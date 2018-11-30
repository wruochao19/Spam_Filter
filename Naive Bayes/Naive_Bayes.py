#! user/bin/pycharmPoject

import numpy as np
import re


# segment text


def text_segment(large_string):
    """
     Desc: Receive a large string and segment it to a string list
     :param large_string:
     :return: String list and turn all string into lower case
    """
    clean_string = re.compile(r'[^a-zA-z]|\d')
    list_string = clean_string.split(large_string)
    list_string_lower = [string.lower() for string in list_string if len(list_string) > 0]
    return list_string_lower


def loadData(filename):
    """
     Desc: Load date from csv file
     :param filename:
     :return: A list marked whether each mail is spam using 1 or 0. And a list contains content of these mails
    """
    category = []
    textWords = []
    import csv
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row['v1'])
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
        vocableList = list(vocable_set)
    return vocableList


def wordsFecOfVoc(vocableList, inputList):
    """
     Desc: find out whether the words in inputList appear in vocable list
     :param vocableList:
     :param inputList:
     :return: the resList, if one word is appeared the corresponding index of the word in resList is 1, else is 0
    """
    resList = [0] * len(vocableList)
    for word in inputList:
        if word in vocableList:
            resList[vocableList.index(word)] += 1
    return resList


def wordsFecListOfVoc(vocableList, wordsList):
    """
     Desc: mark a matrix of vocableList with tag that is sum of occurrences of words
     :param vocableList:
     :param wordsList:
     :return: a matrix of occurrences of words in vocableList
    """
    setOfwordList = []
    for i in range(len(wordsList)):
        # print(i)
        setOfword = wordsFecOfVoc(vocableList, wordsList[i])
        setOfwordList.append(setOfword)
    return setOfwordList


def trainNB(trainMatrix, trainCategory):
    """
     Desc: calculate probability of spam, probability that words will appear in the spam and ham
     :param trainMatrix: marked matrix of occurences of words.
     :param trainCategory:  category of email as spam and ham
     :return: probability
     """
    numOfmails = len(trainMatrix)
    numOfwords = len(trainMatrix[0])
    # P(s)
    print(sum(trainCategory))
    print(numOfmails)
    Pspam = sum(trainCategory) / float(numOfmails)
    # initial each word as 1 in mails
    WordsFecOfSpam = np.ones(numOfwords)
    WordsFecOfHam = np.ones(numOfwords)
    numWordsInSpam = 2.0
    numWordsInHam = 2.0
    for i in range(0, numOfmails):
        if trainCategory[i] == 1:
            WordsFecOfSpam += trainMatrix[i]
            numWordsInSpam += sum(trainMatrix[i])
        else:
            WordsFecOfHam += trainMatrix[i]
            numWordsInHam += sum(trainMatrix[i])
    PwordsOfSpam = np.log(WordsFecOfSpam / numWordsInSpam)
    PwordsOfHam = np.log(WordsFecOfHam / numWordsInHam)
    return PwordsOfSpam, PwordsOfHam, Pspam


def classify(VocableList, PwordsOfSpam, PwordsOFham, Pspam, target_wrods):
    """
     :param VocableList:
     :param PwordsOfSpam:
     :param PwordsOFham:
     :param Pspam:
     :param target_wrods:
     :return:
    """
    PwordsOfSpam = PwordsOfSpam[1:]
    PwordsOFham = PwordsOFham[1:]
    words_count = wordsFecOfVoc(VocableList, target_wrods)
    ArrayOfwords = np.array(words_count)
    p1 = sum(ArrayOfwords * PwordsOfSpam) + np.log(Pspam)

    p0 = sum(ArrayOfwords * PwordsOFham) + np.log(1 - Pspam)

    if p1 > p0:
        return 1
    else:
        return 0


def getVocableList(file_name):
    fw = open(file_name)
    vocableList = fw.readline().strip().split('\t')
    fw.close()
    return vocableList


def getTrainInfo():
    vocableList = getVocableList('vocableList.txt')
    pWordsOfSpam = np.loadtxt('pWordsOfSpam.txt', delimiter='\t')
    pWordsOfHam = np.loadtxt('pWordsOfHam.txt', delimiter='\t')
    fr = open('pSpam.txt')
    pSpam = float(fr.readline().strip())
    fr.close()
    return vocableList, pWordsOfSpam, pWordsOfHam, pSpam
