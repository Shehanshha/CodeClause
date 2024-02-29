import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy

nltk.download('wordnet')
from tensorflow.keras.models import load_model
model = load_model('ChatBot-using-Keras-TkinterGUI-main\chatbot_model.h5')
import json
import random
from keras.models import load_model
model = load_model('ChatBot-using-Keras-TkinterGUI-main\chatbot_model.h5')

intents = json.loads(open('ChatBot-using-Keras-TkinterGUI-main\intents.json').read())
words = pickle.load(open('ChatBot-using-Keras-TkinterGUI-main\words.pkl','rb'))
classes = pickle.load(open('ChatBot-using-Keras-TkinterGUI-main\classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result  
    return "Sorry, I didn't understand that." 


def chatbot_response(msg):
    ints = predict_class(msg, model)
    print(ints)  # Print the predicted intents and their probabilities
    res = getResponse(ints, intents)
    return res

import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#000000", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("Chatbot")
base.geometry("500x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="100", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#cdf725", activebackground="#3c9d9b",fg='#000000',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")


#Place all components on the screen
scrollbar.place(x=480,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=500)
EntryBox.place(x=128, y=401, height=90, width=500)
SendButton.place(x=6, y=401, height=90)

base.mainloop()