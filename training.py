import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        word_list = [word for word in word_list if word not in stop_words]
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = [pair[0] for pair in training]
train_y = [pair[1] for pair in training]

# Normalización de datos
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

# Dividir el conjunto de datos en entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=400, alpha=1e-6,
                    solver='adam', verbose=1, learning_rate_init=0.001)

mlp.fit(x_train, y_train)

# Evaluar el modelo con el conjunto de validación
val_score = mlp.score(x_val, y_val)
print(f"Validation Score: {val_score}")

pickle.dump(mlp, open("chatbot_model.pkl", "wb"))
