from nltk.tag import pos_tag
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re, string,random
import pandas as pd
import pickle
import pymorphy2

#Устранение шума, нормализация
def removeNoise(stemmer, tweetTokens,stopWords=()):
    cleaned_tokens=[]
    for token, tag in pos_tag(tweetTokens, lang='rus'): #Определение тегов
        #Удаление гиперссылок и пользователей
        token=re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
        '(?:%[0-9a-zA-Z][0-9a-zA-Z]))+','', token)
        token=re.sub("(@[A-Za-z0-9_]+)","", token)
        token = re.sub("RT", "", token)
        token = re.sub("(\.{2,})", "", token)

        if token not in string.punctuation and tag!="NONLEX":
            if tag!="NONLEX":
                token=stemmer.parse(token)[0].normal_form
                token = re.sub("(^это$|^весь$|^ещё$)", "", token)
            if len(token)>0  and token.lower() not in stopWords:
                cleaned_tokens.append(token.lower())
    return cleaned_tokens



#Получение форматированного списка токенов
def getAllWords(cleanedTokensList):
    for tokens in cleanedTokensList:
        for token in tokens:
            yield token

#Создание словаря
def getTweetsForModel(cleanedTokensList, wordFeatures):
    for tweetTokens in cleanedTokensList:
        yield dict([token,token in wordFeatures]for token in tweetTokens)




def __main__():
    # Загрузка данных их файла
    file_pos = pd.read_csv('positive.csv', sep=';')
    file_neg = pd.read_csv('negative.csv', sep=';')

    # Получение набора данных для разных типов твитов
    positiveTweets = file_pos.iloc[:, 3].tolist().copy()
    negativeTweets = file_neg.iloc[:, 3].tolist().copy()

    # Разобивка на токены
    tknzr = TweetTokenizer()
    positiveTweetTokens = [tknzr.tokenize(tweet) for tweet in positiveTweets]
    negativeTweetTokens = [tknzr.tokenize(tweet) for tweet in negativeTweets]
    stop_words = stopwords.words("russian")  # Загрузка стоп слов
    print(positiveTweetTokens[500])
    print(negativeTweetTokens[500])

    # Убирание шума, нормализация
    stemmer = pymorphy2.MorphAnalyzer()
    positiveCleanedTokensList = []
    negativeCleanedTokensList = []
    for tokens in positiveTweetTokens:
        positiveCleanedTokensList.append(removeNoise(stemmer, tokens, stop_words))
    print(positiveCleanedTokensList[500])
    for tokens in negativeTweetTokens:
        negativeCleanedTokensList.append(removeNoise(stemmer, tokens, stop_words))
    print(negativeCleanedTokensList[500])

    # Получение частоты встречаемости
    allPosWords = getAllWords(positiveCleanedTokensList)
    freqDistPos = FreqDist(allPosWords)
    wordFeaturesPos = list(freqDistPos)[:4000]
    print(list(freqDistPos)[:10])
    allNegWords = getAllWords(negativeCleanedTokensList)
    freqDistNeg = FreqDist(allNegWords)
    wordFeaturesNeg = list(freqDistNeg)[:4000]
    print(list(freqDistNeg)[:10])

    # Создание словарей
    positiveTokensForModel = getTweetsForModel(positiveCleanedTokensList, wordFeaturesPos)
    negativeTokensForModel = getTweetsForModel(negativeCleanedTokensList, wordFeaturesNeg)

    # Создание обучающей и тестовой выборки
    positiveDataset = [(tweetDict, "Positive") for tweetDict in positiveTokensForModel]
    negativeDataset = [(tweetDict, "Negative") for tweetDict in negativeTokensForModel]
    dataset = positiveDataset + negativeDataset
    random.shuffle(dataset)  # Смешивание
    knife= int((len(dataset)*7)/10)
    trainData = dataset[:knife]  # Разбивание в отношении 70/30
    testData = dataset[knife:]

    # Создание, обучение и тестирования модели
    classifier = NaiveBayesClassifier.train(trainData)
    print("Accuracy is:", classify.accuracy(classifier, testData))
    print(classifier.show_most_informative_features(10))

    f = open('classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

if __name__ == "__main__":
    __main__()