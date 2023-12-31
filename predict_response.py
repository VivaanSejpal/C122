#Text Data Preprocessing Lib



    # """
    # The code is a Python implementation of a chatbot using a pre-trained model to generate responses
    # based on user input.
    
    # :param user_input: The user_input parameter is the input message provided by the user. It is the
    # text that the user wants to communicate with the chatbot
    # :return: The code returns the response generated by the chatbot based on the user input.
    # """

# The code is importing necessary libraries and modules for the chatbot implementation.
import nltk
import json
import pickle
import numpy as np
import random

# The `ignore_words` list is used to filter out certain words from the user input before processing
# it. These words are typically punctuation marks and contractions that do not carry much meaning and
# can be safely ignored during text preprocessing. In this case, the list includes punctuation marks
# like `?`, `!`, `,`, and `.` as well as contractions like `'s` and `'m`. These words will not be
# considered when tokenizing and encoding the user input.
ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib
# The code is importing the TensorFlow library and a function called `get_stem_words` from a module
# called `data_preprocessing`.
import tensorflow
from data_preprocessing import get_stem_words

# The line `model = tensorflow.keras.models.load_model('./chatbot_model.h5')` is loading a pre-trained
# model for the chatbot. The model is stored in a file called `chatbot_model.h5` and it is being
# loaded using the `load_model` function from the `tensorflow.keras.models` module. Once the model is
# loaded, it can be used to make predictions on user input.
model = tensorflow.keras.models.load_model('/Users/viralsejpal/Downloads/PRO-C120-Teacher-Boilerplate-Code-main/static/assets/model_files/chatbot_model.h5')

# Load data files
# The code is loading data files that are necessary for the chatbot to function properly.
intents = json.loads(open('/Users/viralsejpal/Downloads/PRO-C120-Teacher-Boilerplate-Code-main/intents.json').read())
words = pickle.load(open('/Users/viralsejpal/Downloads/PRO-C120-Teacher-Boilerplate-Code-main/static/assets/chatbot_corpus/words.pklhi','rb'))
classes = pickle.load(open('/Users/viralsejpal/Downloads/PRO-C120-Teacher-Boilerplate-Code-main/static/assets/chatbot_corpus/classes.pkl','rb'))


def preprocess_user_input(user_input):
    """
    The function preprocesses user input by tokenizing the input, stemming the words, creating a bag of
    words representation, and encoding the input data.
    
    :param user_input: The user_input parameter is the input provided by the user, which is a string of
    text that needs to be preprocessed
    :return: a numpy array containing the bag of words representation of the preprocessed user input.
    """

    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []
   
    # Input data encoding 
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
  
    return np.array(bag)

def bot_class_prediction(user_input):
    """
    The function `bot_class_prediction` takes user input, preprocesses it, and uses a model to predict
    the class label for the input.
    
    :param user_input: The user_input parameter is the input provided by the user, which could be a
    sentence, a question, or any other text input
    :return: the predicted class label for the user input.
    """

    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label


def bot_response(user_input):
    """
    The function `bot_response` takes user input, predicts the class label using `bot_class_prediction`,
    and returns a random response from the corresponding intent in the `intents` dictionary.
    
    :param user_input: The user's input, which is the text that the user has entered as their message or
    query
    :return: The bot_response variable is being returned.
    """
    predicted_class_label =  bot_class_prediction(user_input)
    predicted_class = classes[predicted_class_label]

    for intent in intents['intents']:
     if intent['tag']==predicted_class:
         bot_response = random.choice(intent['responses'])
         return bot_response

print("Hi I am Stella, How Can I help you?")

# The code block `while True:` is creating an infinite loop that allows the chatbot to continuously
# interact with the user.
while True:
    user_input = input("Type your message here:")
    print("User Input: ", user_input)

    response = bot_response(user_input)
    print("Bot Response: ", response)
