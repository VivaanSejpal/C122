# The code `from flask import Flask, render_template, url_for, request, jsonify` is importing the
# necessary modules from the Flask library.
from flask import Flask, render_template, url_for, request, jsonify
from model_prediction import * 
from predict_response import *
 
# `app = Flask(__name__)` creates an instance of the Flask class and assigns it to the variable `app`.
# The `__name__` argument is a special Python variable that represents the name of the current module.
# By passing `__name__` as the argument, we are telling Flask to use the current module as the
# starting point of the application.
app = Flask(__name__)

# The lines `predicted_emotion=""` and `predicted_emotion_img_url=""` are initializing two variables
# `predicted_emotion` and `predicted_emotion_img_url` with empty strings. These variables will be used
# to store the predicted emotion and the URL of the image representing the predicted emotion. By
# initializing them with empty strings, we ensure that they have a default value before any prediction
# is made.
predicted_emotion=""
predicted_emotion_img_url=""

# The `@app.route('/')` decorator is defining a route for the Flask application. It specifies that the
# route `/` (root URL) should be associated with the function `index()`. This means that when a GET
# request is made to the root URL, the `index()` function will be executed.
@app.route('/')
def index():
    """
    The function "index" retrieves entries and renders them in the "index.html" template.
    :return: the rendered template "index.html" with the variable "entries" passed as an argument.
    """
    entries = show_entry()
    return render_template("index.html", entries=entries)
 
# The `@app.route('/predict-emotion', methods=["POST"])` decorator is defining a route for the Flask
# application. It specifies that the route `/predict-emotion` should be associated with the function
# `predict_emotion()`. This means that when a POST request is made to the `/predict-emotion` route,
# the `predict_emotion()` function will be executed.
@app.route('/predict-emotion', methods=["POST"])
def predict_emotion():
    """
    The function `predict_emotion()` takes input text from a POST request, predicts the emotion
    associated with the text, and returns the predicted emotion and an image URL representing the
    emotion.
    :return: a JSON response. If the input_text is undefined, it returns an error response with a status
    and message. If the input_text is defined, it returns a success response with the predicted emotion
    and the URL of an image representing the predicted emotion.
    """
    
    # Get Input Text from POST Request
    input_text = request.json.get("text")  
    
    if not input_text:
        # Response to send if the input_text is undefined
        response = {
                    "status": "error",
                    "message": "Please enter some text to predict emotion!"
                  }
        return jsonify(response)
    else:  
        predicted_emotion, predicted_emotion_img_url = predict(input_text)
        
        # Response to send if the input_text is not undefined
        response = {
                    "status": "success",
                    "data": {
                            "predicted_emotion": predicted_emotion,
                            "predicted_emotion_img_url": predicted_emotion_img_url
                            }  
                   }

# The `@app.route("/save-entry", methods=["POST"])` decorator is defining a route for the Flask
# application. It specifies that the route `/save-entry` should be associated with the function
# `save_entry()`. This means that when a POST request is made to the `/save-entry` route, the
# `save_entry()` function will be executed.
        # Send Response         
        return jsonify(response)


@app.route("/save-entry", methods=["POST"])
def save_entry():
    """
    The `save_entry` function takes in a date, predicted emotion, and text entered by the user, and
    saves it as a new entry in a CSV file.
    :return: a JSON response with the message "Success".
    """

    # Get Date, Predicted Emotion & Text Enter by the user to save the entry
    date = request.json.get("date")           
    emotion = request.json.get("emotion")
    save_text = request.json.get("text")

    save_text = save_text.replace("\n", " ")

    # CSV Entry
    entry = f'"{date}","{save_text}","{emotion}"\n'  

    with open("./static/assets/data_files/data_entry.csv", "a") as f:
        f.write(entry)
    return jsonify("Success")


# The `@app.route("/bot-response", methods=["POST"])` decorator is defining a route for the Flask
# application. It specifies that the route `/bot-response` should be associated with the function
# `bot()`. This means that when a POST request is made to the `/bot-response` route, the `bot()`
# function will be executed.
@app.route("/bot-response", methods=["POST"])
def bot():
    """
    The function `bot()` takes user input, calls the `bot_response()` method to generate a response, and
    returns the response as a JSON object.
    :return: The function `bot()` returns a JSON response containing the bot's response to the user's
    input. The bot's response is stored in the "bot_response" field of the JSON object.
    """
    # Get User Input
    input_text = request.json.get("user_bot_input_text")
   
    # Call the method to get bot response
    bot_res = bot_response(input_text)

    response = {
            "bot_response": bot_res
        }

    return jsonify(response)     
     
# The `if __name__ == '__main__':` block is used to ensure that the Flask application is only run when
# the script is executed directly, and not when it is imported as a module.
if __name__ == '__main__':
    app.run(debug=True)