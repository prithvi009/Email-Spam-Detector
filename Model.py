import math

SPAM = 'spam'
HAM = 'ham'
WRONG = 'wrong'
RIGHT = 'right'


class NaiveBayesClassifier:

    def __init__(self):
        self.PriorH = 0.0
        self.PriorS = 0.0
        self.vocabulary = {}
        self.result = {}
        self.S_True_Positive = 0
        self.S_False_Negative = 0
        self.H_True_Negative = 0
        self.H_False_Positive = 0

    def getPriorHam(self):
        return self.PriorH
    
    def setPriorHam(self, totalDocuments, hamDocuments):
        self.PriorH = math.log10(hamDocuments / totalDocuments)

    def getPriorSpam(self):
        return self.PriorS
    
    def setPriorSpam(self, totalDocuments, spamDocuments):
        self.PriorS = math.log10(spamDocuments / totalDocuments)

  
    def fit(self, vocabulary):
        self.vocabulary = vocabulary
    

    def getClassificationResult(self):
        return self.result
    
   
    def addClassificationResult(self, document, predictedClass, scoreHam, scoreSpam, actualClass, label):
        self.result[document] = [predictedClass, scoreHam, scoreSpam, actualClass, label]

   
    def setConfusionMatrixVar(self, Target, Predicted):
        if Target == SPAM and Predicted == SPAM:
            self.S_True_Positive += 1
        elif Target == SPAM and Predicted == HAM:
            self.S_False_Negative += 1
        elif Target == HAM and Predicted == HAM:
            self.H_True_Negative += 1
        elif Target == HAM and Predicted == SPAM:
            self.H_False_Positive += 1

    
    def predict(self, document, actualClass, words):
        scoreHam = self.getPriorHam()
        scoreSpam = self.getPriorSpam()
        predictedClass = ''
        label = ''

        for word in words:
            if word in self.vocabulary:
                hamProb = self.vocabulary[word][1]
                spamProb = self.vocabulary[word][3]
                scoreHam += math.log10(hamProb)
                scoreSpam += math.log10(spamProb)
        
        if scoreHam > scoreSpam:
            predictedClass = HAM
        else:
            predictedClass = SPAM
        
        if predictedClass == actualClass:
            label = RIGHT
        else:
            label = WRONG
        
        self.addClassificationResult(document, predictedClass, scoreHam, scoreSpam, actualClass, label)
        self.setConfusionMatrixVar(actualClass, predictedClass)

    
    def getAccuracy(self):
        total = self.S_True_Positive + self.H_True_Negative + self.S_False_Negative + self.H_False_Positive
        accuracy = (self.S_True_Positive + self.H_True_Negative) / total
        return accuracy

    
    def getPrecision(self):
        precision = self.S_True_Positive / (self.S_True_Positive + self.H_False_Positive)
        return precision
    
    
    def getRecall(self):
        recall = self.S_True_Positive / (self.S_True_Positive + self.S_False_Negative)
        return recall

    
    def getF1Measure(self):
        precision = self.getPrecision()
        recall = self.getRecall()
        f1Mesaure = 2 * ( (precision  * recall) / (precision  + recall) )
        return f1Mesaure
    
    
    def printConfusionMatrix(self):
        TP = str(self.S_True_Positive)
        FN = str(self.S_False_Negative)
        TN = str(self.H_True_Negative)
        FP = str(self.H_False_Positive)
        message = (
            
            "                   +-----------------------+----------------------+" + "\n" +
            "                   |   (Predicted) SPAM    |   (Predicted) HAM    |" + "\n" +
            "+------------------+-----------------------+----------------------+" + "\n" +
            "| (Actual) SPAM    |          "+TP+"          |         "+FN+"           |" + "\n" +
            "+------------------+-----------------------+----------------------+" + "\n" +
            "|  (Actual) HAM    |          "+FP+"            |          "+TN+"         |" + "\n" +
            "+------------------+-----------------------+----------------------+" + "\n" 
        )
        print("          CONFUSION_MATRIX         "+ "\n")
        print(message)

    

    
        



