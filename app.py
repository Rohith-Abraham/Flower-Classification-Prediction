from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # Import the image module
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Create a data generator for loading images
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Specify the directory where your flowers are stored
train_data_dir = "D:\\sk sir\\lab\\flowers"  # Update this path

# Create the train generator
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Now you can use train_generator.class_indices.keys() as needed


app = Flask(__name__)

# Define the path for uploads directory
uploads_dir = os.path.join(app.root_path, 'static', 'uploads')

# Create uploads directory if it doesn't exist
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Load the trained model
model = load_model("D:\\sk sir\\lab\\model.h5")  # Replace with the actual path

# Define the list of flower names manually (based on your model's training)
flower_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']  # Update this list as needed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        # Save the uploaded file to the uploads directory
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Rescale image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Use the manually defined flower names
        predicted_flower = flower_names[predicted_class[0]]

        # Pass the filename and predicted flower name to the result template
        return render_template('result.html', flower=predicted_flower, filename=file.filename)

    return render_template('index.html')


# Route for serving uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)


