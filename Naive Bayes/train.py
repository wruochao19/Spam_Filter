import numpy as np
import Naive_Bayes as naiveBayes

file_name = 'spam.csv'
trainCategory, textWords = naiveBayes.loadData(file_name)
vocableList = naiveBayes.creat_vocable_list(textWords)
wordsFecList = naiveBayes.wordsFecListOfVoc(vocableList, textWords)
trainMatrix = np.array(wordsFecList)
pWordsOfSpam, pWordsOfHam, pSpam = naiveBayes.trainNB(trainMatrix, trainCategory)
file_pSpam = open('pSpam.txt', 'w')
spam = pSpam.__str__()
file_pSpam.write(spam)
file_pSpam.close()
file_word = open('vocableList.txt', 'w')
for i in range(len(vocableList)):
    file_word.write(vocableList[i]+'\t')
file_word.flush()
file_word.close()
np.savetxt('pWordsOfSpam.txt', pWordsOfSpam,delimiter='\t')
np.savetxt('pWordsOfHam.txt', pWordsOfHam, delimiter='\t')