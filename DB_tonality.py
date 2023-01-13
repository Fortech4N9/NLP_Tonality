from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from pymongo import MongoClient


import re, string

def removeNoise(tweetTokens, stopWords = ()):

    cleanedTokens = []

    for token, tag in pos_tag(tweetTokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stopWords:
            cleanedTokens.append(token.lower())
    return cleanedTokens

def getAllWords(cleanedTokensList):
    for tokens in cleanedTokensList:
        for token in tokens:
            yield token

def getTweetsForModel(cleanedTokensList):
    for tweetTokens in cleanedTokensList:
        yield dict([token, True] for token in tweetTokens)



def __main__():
    modelFile = Path('classifier.pickle')
    if not modelFile.exists():
        print("Нужна модель")
    else:
        # Подключение к базе данных
        client = MongoClient('localhost', 27017)
        db = client.NLP
        # in_bd(db)

        # Подключение модели и необходимых элементов анализа
        f = open('classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()

        allNews = db.Second
        cnt = 0
        #Для всех новостей из коллекции
        for news in allNews.find():
            #Получение id записи
            _id = news["_id"]
            print(_id)
            #Получение текста для анализа тональности
            tweet = news["text"]

            string = ""

            endSentence = tweet.find('.')

            startSent = 0

            cnt = 0

            while ('.' in tweet[startSent:]):
                cnt += 1
                sentence = tweet[startSent : endSentence]


                customTokens = []
                customTokens = removeNoise(word_tokenize(sentence))

                #Данные для обновления необходимой записи с найденной тональностью


                data = str({'Тональность': classifier.classify(dict([token, True] for token in customTokens))})
                string = string + str(cnt) + ")" + data[1:-1] + "\n "

                startSent = tweet.find('\n', endSentence) + 1
                endSentence = tweet.find('.', startSent)

                #Обновление записи в БД
            newsId = news['_id']
            print(newsId)

            newCollection = db.Third
            newCollection.find_one_and_delete({'_id': newsId})
            newCollection.insert_one(
                {
                    '_id': newsId,
                    'text': tweet,
                    'ton': string
                }
            )
            cnt += 1

if __name__ == "__main__":
    __main__()