import re
import os
import math


class TextProcessor:

    def __init__(self):
        self.word_frequency = {}
        self.words_Ham = {}
        self.words_Spam = {}
        self.vocabulary = {}
        self.Delta = 0.5
        self.sizeOfHam = 0
        self.sizeOfSpam = 0
        self.sizeOfCorpus = 0
    
   
    def tokenize(self, text):
        return re.split('[^a-zA-Z]', text)

    
    def recordWordCount(self, words):
        for word in words:
            if word != '':
                if word.lower() in self.word_frequency:
                    self.word_frequency[word.lower()] += 1
                else:
                    self.word_frequency[word.lower()] = 1
    
    
    def getWordsFromDocument(self, words):
        wordsList = []
        for word in words:
            if word != '':
                wordsList.append(word.lower())
        return wordsList

    
    def updatefrequencyCountInClass(self, classType, words):
        for word in words:
            if word != '':
                if(classType == 'ham'):
                    if word.lower() in self.words_Ham:
                        self.words_Ham[word.lower()] += 1
                    else:
                        self.words_Ham[word.lower()] = 1
                        
                if(classType == 'spam'):
                    if word.lower() in self.words_Spam:
                        self.words_Spam[word.lower()] += 1
                    else:
                        self.words_Spam[word.lower()] = 1
    
    
    def calculateCondProb(self, frequency, classType):
        freqWord = 0
        sizeOfCorpus = self.sizeOfCorpus
        freqWord = frequency

        if(classType == 'ham'):
            totalNoOfWords = self.sizeOfHam
        else:
            totalNoOfWords = self.sizeOfSpam
        
        prob = ( (freqWord + self.Delta) / (totalNoOfWords + (self.Delta * sizeOfCorpus)) )
        return prob
    
   
    def getWordFrequency(self):
        return self.word_frequency

    
    def getWordsHam(self):
        return self.words_Ham
    
    
    def getWordsSpam(self):
        return self.words_Spam
    
    
    def getVocabulary(self):
        return self.vocabulary
    
    def setVocabulary(self, word, result):
        self.vocabulary[word] = result

    def setFreqHam(self, word, value):
        self.vocabulary.get(word, "")[0] = value

    def setFreqSpam(self, word, value):
        self.vocabulary.get(word, "")[2] = value
        pass

    def setConditinalProbHam(self, word, value):
        self.vocabulary.get(word, "")[1] = value

    def setConditinalProbSpam(self, word, value):
        self.vocabulary.get(word, "")[3] = value
    
    
    def buildVocabulary(self):
        self.sizeOfCorpus = len(self.word_frequency)
        self.sizeOfHam = sum(self.words_Ham.values())
        self.sizeOfSpam = sum(self.words_Spam.values())

        sortedCorpus = sorted(self.word_frequency.keys(), key=lambda x:x.lower())

        for key in sortedCorpus:
            value = self.word_frequency[key]
            self.setVocabulary(key, [0, 0.0, 0, 0.0])

            if key in self.words_Ham:
                self.setFreqHam(key, self.words_Ham[key])
                probability = self.calculateCondProb(self.words_Ham[key], 'ham')
                self.setConditinalProbHam(key, probability)
            else:
                self.setFreqHam(key, 0)
                probability = self.calculateCondProb(0, 'ham')
                self.setConditinalProbHam(key, probability)
                
            if key in self.words_Spam:
                self.setFreqSpam(key, self.words_Spam[key])
                probability = self.calculateCondProb(self.words_Spam[key], 'spam')
                self.setConditinalProbSpam(key, probability)
            else:
                self.setFreqSpam(key, 0)
                probability = self.calculateCondProb(0, 'spam')
                self.setConditinalProbSpam(key, probability)

    

class FileProcessor:

    def __init__(self):
        self.space = "  "
    
    
    def loadDataFiles(self, path):
        return os.listdir(path)

    
    def getDocumentName(self, file):
        docBase = os.path.basename(file.__getattribute__('name'))
        docName = os.path.splitext(docBase)[0]
        return docName

    
    def getClassType(self, file):
        docName = self.getDocumentName(file)

        if re.search('(.*)-(ham)-(.*)', docName):
            return 'ham'
        else:
            return 'spam'

    
    def getNumOfDocuments(self, files):
        totalDocuments = len(files)
        HamDocuments = 0
        SpamDocuments = 0
        for file in files:
            if re.search('(.*)-(ham)-(.*)', file):
                HamDocuments += 1
            else:
                SpamDocuments += 1    

        return totalDocuments, HamDocuments, SpamDocuments
        
    
    def processFiles(self, files, path, textProcessor):
       
        for file in files:
            try:
                with open(str(path+file), "r", encoding="utf8", errors='ignore') as f:
                    classType = self.getClassType(f)

                    for line in f:
                        line = line.strip()
                        words = textProcessor.tokenize(line)
                        textProcessor.recordWordCount(words)
                        textProcessor.updatefrequencyCountInClass(classType, words)

            finally:
                f.close()

    
    def storeVocabulary(self, file, vocabulary):
        try:
            with open(file, "w") as f:
                lineNum = 0
                for key, value in vocabulary.items():
                    lineNum += 1
                    lineString = (str(lineNum) + self.space 
                    + str(key) + self.space
                    + str(value[0]) + self.space 
                    + str(value[1]) + self.space 
                    + str(value[2]) + self.space 
                    + str(value[3]) + "\r")

                    f.write(lineString)

        finally:
            f.close()

    
    def storeClassificationResult(self, file, result):
        try:
            with open(file, "w") as f:
                lineNum = 0
                for key, value in result.items():
                    lineNum += 1
                    lineString = (str(lineNum) + self.space 
                    + str(key) + self.space
                    + str(value[0]) + self.space 
                    + str(value[1]) + self.space 
                    + str(value[2]) + self.space 
                    + str(value[3]) + self.space 
                    + str(value[4]) +"\r")

                    f.write(lineString)

        finally:
            f.close()
   
    

    
