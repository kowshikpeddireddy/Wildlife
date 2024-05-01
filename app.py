import subprocess
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import openai
import os
import mysql.connector
import numpy as np

app = Flask ( __name__,static_folder='static' )

loaded_model = tf.keras.models.load_model ( 'animals.h5' )
class_names = {
    0: 'antelope', 1: 'badger', 2: 'bat', 3: 'bear', 4: 'bee', 5: 'beetle', 6: 'bison', 7: 'boar', 8: 'butterfly',
    9: 'cat', 10: 'caterpillar', 11: 'chimpanzee', 12: 'cockroach', 13: 'cow', 14: 'coyote', 15: 'crab', 16: 'crow',
    17: 'deer', 18: 'dog', 19: 'dolphin', 20: 'donkey', 21: 'dragonfly', 22: 'duck', 23: 'eagle', 24: 'elephant',
    25: 'flamingo', 26: 'fly', 27: 'fox', 28: 'goat', 29: 'goldfish', 30: 'goose', 31: 'gorilla', 32: 'grasshopper',
    33: 'hamster', 34: 'hare', 35: 'hedgehog', 36: 'hippopotamus', 37: 'hornbill', 38: 'horse', 39: 'hummingbird',
    40: 'hyena', 41: 'jellyfish', 42: 'kangaroo', 43: 'koala', 44: 'ladybugs', 45: 'leopard', 46: 'lion',
    47: 'lizard', 48: 'lobster', 49: 'mosquito', 50: 'moth', 51: 'mouse', 52: 'octopus', 53: 'okapi', 54: 'orangutan',
    55: 'otter', 56: 'owl', 57: 'ox', 58: 'oyster', 59: 'panda', 60: 'parrot', 61: 'pelecaniformes', 62: 'penguin',
    63: 'pig', 64: 'pigeon', 65: 'porcupine', 66: 'possum', 67: 'raccoon', 68: 'rat', 69: 'reindeer', 70: 'rhinoceros',
    71: 'sandpiper', 72: 'seahorse', 73: 'seal', 74: 'shark', 75: 'sheep', 76: 'snake', 77: 'sparrow', 78: 'squid',
    79: 'squirrel', 80: 'starfish', 81: 'swan', 82: 'tiger', 83: 'turkey', 84: 'turtle', 85: 'whale', 86: 'wolf',
    87: 'wombat', 88: 'woodpecker', 89: 'zebra'
}

# Set up OpenAI API key
api_key = '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'
openai.api_key = api_key


# Function to preprocess user input image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open ( image_path )
    img = img.resize ( target_size )
    img = np.asarray ( img ) / 255.0  # Normalize the image
    img = np.expand_dims ( img, axis=0 )  # Add batch dimension
    return img


# Function to predict the class name of an image
def predict_class_name(image_path, model):
    img = preprocess_image ( image_path )
    prediction = loaded_model.predict ( img )
    predicted_class_index = np.argmax ( prediction )
    predicted_class_name = class_names.get ( predicted_class_index, 'Unknown' )
    return predicted_class_name


# Function to get details about the animal from ChatGPT
def get_animal_details(animal_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"What is the average lifespan of a {animal_name}?"},
        {"role": "user", "content": f"What is the typical diet of a {animal_name}?"},
        {"role": "user", "content": f"Where is {animal_name} mostly found?"},
        {"role": "user", "content": f"Provide a detailed description of a {animal_name}."},
    ]
    try:
        response = openai.ChatCompletion.create (
            model="gpt-3.5-turbo",
            messages=messages
        )
        if 'choices' in response and response['choices']:
            animal_details = [choice['message']['content'] for choice in response['choices']]
            return ' '.join ( animal_details )  # Concatenate all details into a single string
        else:
            return "Error: Empty response received from API"
    except Exception as e:
        return f"Error during API request: {e}"
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/stats')
def stats():
    return render_template('stats.html')
@app.route('/main', methods=['GET', 'POST'])
def predict_animal():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']

        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        # If file exists and is allowed file type
        if file:
            # Save the file to the 'static' directory
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            # Predict the class name of the uploaded image
            predicted_class_name = predict_class_name(file_path, loaded_model)

            # Get details about the predicted animal
            animal_details = get_animal_details(predicted_class_name)

            # Render the template with results
            return render_template('main.html', image=file_path, predicted_class=predicted_class_name,
                                   details=animal_details)

    return render_template('main.html')
# Load the saved model
model = tf.keras.models.load_model('fires.h5')

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Function to make predictions on the loaded image
def predict_image(image):
    # Load and preprocess the image
    img = load_and_preprocess_image(image)
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))
    # Get the predicted class label
    predicted_class = np.argmax(prediction, axis=1)[0]

    return predicted_class


@app.route ('/fires', methods=['GET', 'POST'] )
def ind():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template ( 'index.html', prediction_message="No file part" )
        file = request.files['file']
        if file.filename == '':
            return render_template ( 'index.html', prediction_message="No selected file" )
        if file:
            # Save the file to the 'static' directory
            file_path = os.path.join ( 'static', file.filename )
            file.save ( file_path )

            # Predict the class index of the uploaded image
            predicted_class_index = predict_image( file_path )

            # Determine if it is a forest fire or non-forest fire based on the predicted class index
            if predicted_class_index == 0:  # Forest Fire
                predicted_fire_name = "Forest Fire"
                # Additional information about forest fires
                fire_details = """
                A forest fire is a destructive phenomenon, characterized by large and uncontrolled burning of forested areas. 
                Some common reasons for forest fires include:
                - Natural causes like lightning
                - Human activities such as campfires, discarded cigarettes, or arson
                - Drought and dry conditions can also contribute to fire spread

                Precautions to prevent forest fires include:
                - Following campfire regulations
                - Properly disposing of cigarettes and other flammable materials
                - Creating fire breaks and maintaining defensible spaces around homes

                Immediate actions to take during a forest fire:
                - Evacuate the area if advised by authorities
                - Call emergency services
                - Follow evacuation routes and stay updated with local news

                """
            else:  # Non-Forest Fire
                predicted_fire_name = "Non-Forest Fire"
                # Additional information about greenery
                fire_details = """
                Greenery refers to the presence of lush vegetation and trees, which are essential components of healthy ecosystems. 
                Greenery plays a vital role in:
                - Providing oxygen through photosynthesis
                - Absorbing carbon dioxide and mitigating climate change
                - Providing habitats for wildlife
                - Improving air quality and reducing pollution

                Preserving greenery is important for maintaining biodiversity and supporting ecological balance.
                """

            return render_template ( 'fires.html', prediction_message=f"Predicted: {predicted_fire_name}",
                                     image=file_path,
                                     fire_details=fire_details )
    return render_template ( 'fires.html', prediction_message=None )

def get_state_data(state):
    connection = None
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host='localhost',
            database='wildlife',
            user='root',
            password='6303'
        )

        if connection.is_connected():
            # Fetch data based on the selected state
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM WildlifeSanctuaries WHERE stateName = %s", (state,))
            row = cursor.fetchone()

            # Format the data
            if row:
                return row
            else:
                return None

    except Exception as e:
        print("Error while connecting to MySQL:", e)
    finally:
        # Close database connection
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/details')
def details():
    return render_template('one.html')

@app.route('/state_data', methods=['POST'])
def state_data():
    state = request.form['state']
    state_data = get_state_data(state)
    return render_template('state_data.html', state_data=state_data)

model2 = tf.keras.models.load_model ( "audio_classification_model.h5" )
classes = ['cat', 'dog']


# Helper function to process audio file
def process_audio(file):
    audio = file.read ()
    audio, _ = tf.audio.decode_wav ( audio, desired_channels=1, desired_samples=16000 )
    audio = tf.squeeze ( audio, axis=-1 )
    audio = tf.expand_dims ( audio, axis=0 )
    return audio


# Helper function to predict class label
def predict_label(audio):
    spectrogram = tf.signal.stft ( audio, frame_length=255, frame_step=128 )
    spectrogram = tf.abs ( spectrogram )
    spectrogram = spectrogram[..., tf.newaxis]
    prediction = model2.predict ( spectrogram )
    predicted_label_index = np.argmax ( prediction, axis=1 )[0]
    return classes[predicted_label_index]


# Route to upload page
@app.route ( '/upload' )
def upload_file():
    return render_template ( 'upload.html' )


# Route to handle file upload and prediction
@app.route ( '/pre', methods=['POST'] )
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        try:
            audio = process_audio ( file )
            predicted_class = predict_label ( audio )
            return render_template('result.html', predicted_class=predicted_class)
        except Exception as e:
            return f"Error: {str ( e )}"

@app.route('/chatbot')
def chatbot():
    try:
        # Call the chatbot code as a subprocess
        subprocess.Popen(['python', 'chatbot.py'])
        return render_template('chatbot.html')
    except Exception as e:
        return "Error launching chatbot: " + str(e)
if __name__ == "__main__":
    app.run(debug=True)


