import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

model = load_model('image_classifier_improved.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=r'c:\Users\Владимир\Desktop\AI 2.0\dataset',
    target_size=(128, 128),
    batch_size=1,
    class_mode=None,  
    shuffle=False
)

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()  

class_indices = test_generator.class_indices  # {'cats': 0, 'dogs': 1}
idx_to_class = {v: k for k, v in class_indices.items()}

filenames = test_generator.filenames
for i in range(len(filenames)):
    print(f"{filenames[i]} => {idx_to_class[predicted_classes[i]]}")
