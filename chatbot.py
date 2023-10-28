import random
import json
import pickle
import joblib
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier

lemmatizer = WordNetLemmatizer()

filename = 'questions_log.txt'


intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = joblib.load('chatbot_model.pkl')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict_proba([bow])[0]
    thresh = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > thresh]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    try:
        return return_list[0]['intent']
    except:
        return None

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    return result
print("Hola!  Hazme una pregunta!")
while True:
    message = input("")

    ints = predict_class(message)
    if ints:
        res = get_response(ints, intents)
        print(">> "  + res)
        print("")
    else:
        res = "No entendí esa parte. ¿Puedes reformularlo?"
        print(">> " + res)
    with open(filename, 'a') as f:
        f.write('Q:' + message + '\n')
        f.write('A:' + res + '\n')
