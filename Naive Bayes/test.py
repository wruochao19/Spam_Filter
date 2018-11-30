
import Naive_Bayes as naiveBayes
import random
import numpy as np


def sample():
    vocableList, pWordsOfSpam, pWordsOfHam, pSpam = naiveBayes.getTrainInfo()
    filename = 'test.csv'
    category, textWords = naiveBayes.loadData(filename)
    mileType = naiveBayes.classify(vocableList, pWordsOfSpam, pWordsOfHam, pSpam, textWords[1])
    print(mileType)

#simpleTest()

def classfiyResult():
    filename = 'spam.csv'
    trainCategory, textWords = naiveBayes.loadData(filename)
    mailNum = 1000
    true_mailType = []
    mailContent =[]
    for i in range(mailNum):
        sample = int(random.uniform(0,len(textWords)))
        true_mailType.append(trainCategory[sample])
        mailContent.append(textWords[sample])
        del trainCategory[sample]
        del textWords[sample]
    vocableList, pWordsOfSpam, pWordsOfHam, pSpam = naiveBayes.getTrainInfo()
    errorCount = 0.0
    for i in range(mailNum):
        mailType = naiveBayes.classify(vocableList,pWordsOfSpam, pWordsOfHam, pSpam, mailContent[i])
        print('forecast', mailType, 'true', true_mailType[i])
        if mailType != true_mailType[i]:
            errorCount += 1
    print('errorNum', errorCount, 'errorRate', errorCount/mailNum)


classfiyResult()