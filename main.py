# !pip install matplotlib
# !pip install numpy 
# !pip install tensorflow 
# !pip install opencv-python 
import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv
from tensorflow.keras import datasets, layers, models
import os

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255


class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

'''
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()
'''
''' #CODE FOR TRAINING THE MODEL INTO image_classifier.keras
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

model.save('image_classifier.keras')
'''

model = models.load_model('image_classifier.keras')

def preprocess_image(image_path):
    """
    Load and preprocess an image for the model.
    Resizes image to 32x32 and normalizes pixel values.
    """
    # Read the image
    img = cv.imread(image_path)
    
    # Convert BGR to RGB (OpenCV reads as BGR, but we need RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Resize to 32x32
    img = cv.resize(img, (32, 32))
    
    # Normalize pixel values (0-255 to 0-1)
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(image_path, model):
    """
    Predict the class of an image using the trained model.
    """
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    return predicted_class, confidence, processed_img[0]

def test_custom_images(image_folder="test_images"):
    """
    Test the model with images from a folder.
    Create a folder called 'test_images' and put your 6 images there.
    """
    # Check if the folder exists
    if not os.path.exists(image_folder):
        print(f"Folder '{image_folder}' not found. Please create it and add your images.")
        return
    
    # Get all image files from the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        print(f"No image files found in '{image_folder}' folder.")
        return
    
    print(f"Found {len(image_files)} images to test.")
    print("-" * 50)
    
    # Test each image
    for i, image_path in enumerate(image_files):
        try:
            predicted_class, confidence, processed_img = predict_image(image_path, model)
            
            print(f"Image {i+1}: {os.path.basename(image_path)}")
            print(f"Predicted: {class_names[predicted_class]}")
            print(f"Confidence: {confidence:.2%}")
            print()
            
            # Display the image
            plt.figure(figsize=(8, 4))
            
            # Original image (resized for display)
            plt.subplot(1, 2, 1)
            plt.imshow(processed_img)
            plt.title(f"Input Image (32x32)")
            plt.axis('off')
            
            # Prediction result
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

# Example usage:
test_custom_images()

# Or test a single image:
# image_path = "path/to/your/image.jpg"
# predicted_class, confidence, processed_img = predict_image(image_path, model)
# print(f"Predicted: {class_names[predicted_class]}")
# print(f"Confidence: {confidence:.2%}")
