import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tflearn
import tensorflow 
import random
import json
# Since the preprocessing of the data and training wilol take a little bit of time,we dont want to wait each time we use the model
# To solve this we will use the module pickle
import pickle

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    # Lets extract the data from our json file
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    # Now lets loop through the JSON data and extract the data we want
    # We will turn the patterns into a list of word using nltk.word_tokenizer rather than having them string
    # We will add the pattern into docs_x and the asscoiated tags into docs_y
    for intent in data['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            # Lets add to our words list 
            words.extend(w)
            docs_x.append(w)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent['tag'])

    # Next we will use word stemming to reduce the vocabulary of our model and attempt to find the more general meaning behind the sentences
    # For example the word "going" might have the stem of 'go'
    # This is done so as to find the more general meaning behind sentences

    # This is done to stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    # lets sort the labels
    labels = sorted(labels)

    # Now we need to create a bag of words as our neural network algorithm require numerical inputs

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                # The word exists
                bag.append(1)
            else:
                # The word does not exist
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle","wb") as f:
          pickle.dump((words,labels,training,output), f)

# Now lets build our model 
# This is a super simple model without ab=nt hyperparameter tuning and is running for 1500 epochs

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7: 
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I didn't get that,try again.")

chat()