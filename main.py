from Processor import TextProcessor
from Processor import FileProcessor
from Model import NaiveBayesClassifier

TRAIN_DOCUMENTS = "dataset/train/"
TEST_DOCUMEMENTS = "dataset/test/"
VOCABULARY_DOCUMENT = "results/model.txt"
RESULT_DOCUMENT = "results/result.txt"

class Console:
    def log(self, text):
        print(str(text)+"...")

def main():
    console = Console()
    textProcessor = TextProcessor()
    fileProcessor = FileProcessor()
    

    console.log("loading train files")
    trainFiles = fileProcessor.loadDataFiles(TRAIN_DOCUMENTS)
    console.log("loading test files")
    testFiles = fileProcessor.loadDataFiles(TEST_DOCUMEMENTS)
    

    console.log("processing train documents")
    fileProcessor.processFiles(trainFiles, TRAIN_DOCUMENTS, textProcessor)


    console.log("building vocabulary")
    textProcessor.buildVocabulary()

    console.log("storing the vocabulary in "+VOCABULARY_DOCUMENT)
    fileProcessor.storeVocabulary(VOCABULARY_DOCUMENT, textProcessor.getVocabulary())

    totalTrainDocs, totalHamDocs, totalSpamDocs = fileProcessor.getNumOfDocuments(trainFiles)


    naiveBayesClassifier = NaiveBayesClassifier()
    naiveBayesClassifier.fit(textProcessor.getVocabulary())
    naiveBayesClassifier.setPriorHam(totalTrainDocs, totalHamDocs)
    naiveBayesClassifier.setPriorSpam(totalTrainDocs, totalSpamDocs)


    console.log("running the classifier on test documents")
    for file in testFiles:
            try:
                with open(str(TEST_DOCUMEMENTS+file), "r", encoding="utf8", errors='ignore') as f:
                    classType = fileProcessor.getClassType(f)
                    wordsList = []

                    for line in f:
                        line = line.strip()
                        wordsList.extend(textProcessor.getWordsFromDocument(textProcessor.tokenize(line)))
                    
                    naiveBayesClassifier.predict(file, classType, wordsList)

            finally:
                f.close()
 

    fileProcessor.storeClassificationResult(RESULT_DOCUMENT, naiveBayesClassifier.getClassificationResult())
    console.log("\nclassification done, result stored at "+RESULT_DOCUMENT)

    
    console.log("printing the perfomance measures")
    naiveBayesClassifier.printConfusionMatrix()
    print("Accuracy measure:  "+ str(naiveBayesClassifier.getAccuracy())+ "\n" )
    print("Precision measure: "+ str(naiveBayesClassifier.getPrecision())+ "\n"  )
    print("recall measure:    "+ str(naiveBayesClassifier.getRecall())+ "\n"  )
    print("f1-measure:        "+ str(naiveBayesClassifier.getF1Measure())+ "\n"  )

if __name__ == "__main__":
    main()
