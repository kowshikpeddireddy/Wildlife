import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
# Load the saved model
loaded_model = tf.keras.models.load_model('animals.h5')

# Define a dictionary to map class indices to class names
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

# Function to preprocess user input image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.asarray(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the class index of an image
def predict_class_index(image_path, model):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index

# Function to predict the class name of an image
def predict_class_name(image_path, model):
    predicted_class_index = predict_class_index(image_path, model)
    predicted_class_name = class_names.get(predicted_class_index, 'Unknown')
    return predicted_class_name

# Define the path to the user input image
user_image_path = 'download (1).jpeg'

# Predict the class name of the user input image
predicted_class_name = predict_class_name(user_image_path, loaded_model)

print("Predicted class:", predicted_class_name)

# Display the user input image and predicted class
user_image = Image.open(user_image_path)
plt.imshow(user_image)
plt.title('Predicted Class: ' + predicted_class_name)
plt.axis('off')
plt.show()
import openai

# Set up OpenAI API key
api_key = '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'
openai.api_key = api_key

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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        if 'choices' in response and response['choices']:
            animal_details = [choice['message']['content'] for choice in response['choices']]
            return ' '.join(animal_details)  # Concatenate all details into a single string
        else:
            return "Error: Empty response received from API"
    except Exception as e:
        return f"Error during API request: {e}"

# Get details about the predicted animal from ChatGPT
animal_details = get_animal_details(predicted_class_name)

print("Details about", predicted_class_name + ":")
print(animal_details)
