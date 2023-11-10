# Text Data Preprocessing Lib
import nltk
# The code is importing necessary libraries and modules for text data preprocessing.
# The code is importing the `PorterStemmer` class from the `nltk.stem` module and creating an instance
# of it called `stemmer`. The `PorterStemmer` class is used for stemming words, which is the process
# of reducing words to their base or root form. In this case, the stemmer will be used to stem the
# words in the text data during the preprocessing step.
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json

# The `import pickle` statement is importing the `pickle` module in Python. `pickle` is a module that
# provides a way to serialize and deserialize Python objects. It allows you to convert complex
# objects, such as lists or dictionaries, into a stream of bytes that can be stored in a file or
# transferred over a network. In this code, the `pickle` module is used to save the `stem_words` and
# `tag_classes` variables as binary files using the `pickle.dump()` function. These files can later be
# loaded and used to preprocess new data without having to recompute the stem words and tag classes.
import pickle

import numpy as np


# The code is initializing two empty lists, `words` and `classes`.
words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data

#list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')
# The code is initializing an empty list called `pattern_word_tags_list` and a list called
# `ignore_words`.
pattern_word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]


# The code is reading the contents of the file 'intents.json' and storing it in the variable
# `train_data_file`. Then, it uses the `json.loads()` function to parse the JSON data from
# `train_data_file` and store it in the variable `intents`. This allows the code to access the intents
# and patterns defined in the JSON file for further processing.
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

def get_stem_words(words, ignore_words):
    """
    The function `get_stem_words` takes a list of words and a list of ignore words, and returns a list
    of stemmed words after removing the ignore words.
    
    :param words: The `words` parameter is a list of words that you want to stem. These words will be
    processed to find their stems
    :param ignore_words: The `ignore_words` parameter is a list of words that should be ignored and not
    included in the `stem_words` list
    :return: a list of stemmed words.
    """
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):
    """
    The function `create_bot_corpus` takes in a list of words, a list of classes, a list of
    pattern-word-tag tuples, and a list of ignore words, and returns the stem words, classes, and
    pattern-word-tag list.
    
    :param words: A list that will store all the words from the patterns in the intents
    :param classes: The `classes` parameter is a list that stores all the unique tags or categories of
    the intents in the bot's corpus. Each intent has a tag associated with it, which helps in
    identifying the purpose or topic of the user's input
    :param pattern_word_tags_list: The parameter "pattern_word_tags_list" is a list that stores tuples
    of pattern words and their corresponding tags. Each tuple represents a pattern and its associated
    intent tag
    :param ignore_words: The `ignore_words` parameter is a list of words that should be ignored or
    excluded from the corpus. These words are typically common words or stop words that do not carry
    much meaning or significance in the context of the bot's responses
    :return: three values: stem_words, classes, and pattern_word_tags_list.
    """
    for intent in intents['intents']:
        # Add all patterns and tags to a list
        # The code is iterating over the patterns in each intent in the `intents` list. For each
        # pattern, it tokenizes the pattern into individual words using the `nltk.word_tokenize()`
        # function. The resulting list of words is then added to the `words` list using the `extend()`
        # method. Additionally, a tuple containing the pattern words and the corresponding intent tag
        # is appended to the `pattern_word_tags_list`. This process is done to collect all the unique
        # words in the patterns and create a list of tuples representing the pattern words and their
        # associated intent tags.
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                        
            pattern_word_tags_list.append((pattern_word, intent['tag']))
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
    stem_words = get_stem_words(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words, classes, pattern_word_tags_list


# Training Dataset: 
# Input Text----> as Bag of Words 
# Tags-----------> as Label

def bag_of_words_encoding(stem_words, pattern_word_tags_list):  
    """
    The function `bag_of_words_encoding` takes a list of stem words and a list of pattern word tags, and
    returns a bag of words encoding for each pattern word tag.
    
    :param stem_words: The `stem_words` parameter is a list of words that have been stemmed. Stemming is
    the process of reducing words to their base or root form. For example, the words "running", "runs",
    and "ran" would all be stemmed to "run"
    :param pattern_word_tags_list: The parameter "pattern_word_tags_list" is a list of tuples. Each
    tuple contains two elements: the first element is a list of words, and the second element is a tag
    associated with those words
    :return: a numpy array called "bag" which contains the bag-of-words encoding for each pattern in the
    pattern_word_tags_list.
    """
    bag = []
    for word_tags in pattern_word_tags_list:

        pattern_words = word_tags[0] 
        bag_of_words = []
        stem_pattern_words= get_stem_words(pattern_words, ignore_words)
        for word in stem_words:            
            if word in stem_pattern_words:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        bag.append(bag_of_words)
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    """
    The function `class_label_encoding` takes a list of classes and a list of word-tag pairs, and
    returns a numpy array of label encodings for each word-tag pair.
    
    :param classes: A list of all the possible classes or tags that can be assigned to the patterns
    :param pattern_word_tags_list: The parameter "pattern_word_tags_list" is a list of tuples where each
    tuple contains a word and its corresponding tag. For example, it could look like this:
    :return: a numpy array of label encodings.
    """
    labels = []
    for word_tags in pattern_word_tags_list:

        labels_encoding = list([0]*len(classes)) 
        tag = word_tags[1]
        tag_index = classes.index(tag)
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)
    return np.array(labels)

def preprocess_train_data():
    """
    The function preprocesses training data by creating a bot corpus, encoding words and tags, and
    returning the encoded data.
    :return: The function preprocess_train_data() returns the variables train_x and train_y.
    """
    # The code is calling the `create_bot_corpus()` function to preprocess the training data. This
    # function takes in four parameters: `words`, `classes`, `pattern_word_tags_list`, and
    # `ignore_words`.
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, 
                                            pattern_word_tags_list, ignore_words)
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(tag_classes, open('classes.pkl','wb'))
    # `train_x = bag_of_words_encoding(stem_words, word_tags_list)` is calling the
    # `bag_of_words_encoding()` function to encode the words in the `word_tags_list` into a
    # bag-of-words representation.
    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    # The line `train_y = class_label_encoding(tag_classes, word_tags_list)` is calling the
    # `class_label_encoding()` function to encode the labels for each word-tag pair in the
    # `word_tags_list`.
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    # The line `return train_x, train_y` is returning the variables `train_x` and `train_y` as the
    # output of the function `preprocess_train_data()`. These variables contain the encoded training
    # data that has been preprocessed for training a machine learning model. `train_x` represents the
    # bag-of-words encoding of the input text, while `train_y` represents the label encoding of the
    # corresponding tags.
    return train_x, train_y

# preprocess_train_data()


