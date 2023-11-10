from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimisers import Adam
from data_preprocessing import preprocess_train_data

def train_bot_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape = (len(train_x[0]), ),activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(64,Activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), Activation = "softmax"))

    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    history = model.fit(train_x, train_y, epochs = 200, batch_size = 5, verbose = True)
    model.save("chatbot_model.h5", history)
    print("model created and saved")