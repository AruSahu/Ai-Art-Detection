import cv2
import numpy as np
from django.shortcuts import render
from .forms import ImageForm
import tensorflow as tf
from django.conf import settings

IMAGE_SIZE = (224, 224)

def load_model_with_custom_objects():
    custom_objects = {
        'Flatten': tf.keras.layers.Flatten,
        # Add any other custom layers or functions your model might use
    }
    return tf.keras.models.load_model(settings.MODEL_PATH, custom_objects=custom_objects)

print("This is a pre-check for architecture")
# Load the model
model = load_model_with_custom_objects()

# After loading the model
print("This is a check for architecture")
print(model.summary())

def load_and_preprocess_image(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    return image

def home(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.save()
            image = cv2.imread(img.image.path)
            processed_image = load_and_preprocess_image(image)
            prediction = model.predict(np.expand_dims(processed_image, axis=0))
            is_ai_generated = prediction[0][0] > 0.5
            result = "AI Generated" if is_ai_generated else "Real Image"
            probability = float(prediction[0][0] if is_ai_generated else prediction[0][1])
            return render(request, 'result.html', {
                'result': result,
                'probability': probability*100,
                'image_url': img.image.url
            })
    else:
        form = ImageForm()
    return render(request, 'home.html', {'form': form})