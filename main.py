import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv
from tensorflow.keras import datasets, layers, models
import os

# Load the CIFAR-10 dataset from Keras and normalize the images (values between 0 and 1)
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

# Class names in CIFAR-10 corresponding to label indices
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

'''
# OPTIONAL: Display the first 16 training images with their labels
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()
'''

''' 
# OPTIONAL: Code to train a new model and save it as 'image_classifier.keras'

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels) 
print(f"Loss:{loss}")
print(f"Accuracy:{accuracy}")

model.save('image_classifier.keras')  # Save trained model
'''

# Load a previously trained model
model = models.load_model('image_classifier.keras')

def preprocess_image(image_path):
    """
    Load and preprocess an image for prediction.
    Resizes the image to 32x32, converts color channels, normalizes pixels.
    """
    img = cv.imread(image_path)  # Read image using OpenCV
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
    img = cv.resize(img, (32, 32))  # Resize to match model's input shape
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension for prediction
    return img

def predict_image(image_path, model):
    """
    Predict the class of a single image using the trained CNN.
    Returns predicted label index, confidence score, and processed image.
    """
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction[0])  # Get index of highest probability
    confidence = prediction[0][predicted_class]  # Confidence score for that class
    return predicted_class, confidence, processed_img[0]

def test_custom_images(image_folder="test_images"):
    """
    Test model predictions on all images inside a given folder.
    You should create a folder named 'test_images' and put 6+ images inside.
    """
    if not os.path.exists(image_folder):
        print(f"Folder '{image_folder}' not found. Please create it and add your images.")
        return

    # Supported image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    # Collect all valid image files from the folder
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))

    if not image_files:
        print(f"No image files found in '{image_folder}' folder.")
        return

    print(f"Found {len(image_files)} images to test.")
    print("-" * 50)

    # Run prediction and display results for each image
    for i, image_path in enumerate(image_files):
        try:
            predicted_class, confidence, processed_img = predict_image(image_path, model)

            print(f"Image {i+1}: {os.path.basename(image_path)}")
            print(f"Predicted: {class_names[predicted_class]}")
            print(f"Confidence: {confidence:.2%}")
            print()

            # Display the processed image and prediction side by side
            plt.figure(figsize=(8, 4))

            # Show the 32x32 image fed to the model
            plt.subplot(1, 2, 1)
            plt.imshow(processed_img)
            plt.title(f"Input Image (32x32)")
            plt.axis('off')

            # Show predicted class and confidence
            plt.subplot(1, 2, 2)
            plt.text(0.1, 0.6, f"Prediction: {class_names[predicted_class]}", fontsize=12)
            plt.text(0.1, 0.4, f"Confidence: {confidence:.2%}", fontsize=12)
            plt.axis('off')
            plt.title("Prediction Result")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            print()

# Run the test function on images in the "test_images" folder
test_custom_images()

# OPTIONAL: Test a single image directly
# image_path = "path/to/your/image.jpg"
# predicted_class, confidence, processed_img = predict_image(image_path, model)
# print(f"Predicted: {class_names[predicted_class]}")
# print(f"Confidence: {confidence:.2%}")
