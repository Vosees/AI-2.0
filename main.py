import os
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Путь к датасету
DATASET_PATH = r'c:\Users\Владимир\Desktop\AI 2.0\dataset'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

# Аугментации
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='training',
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='validation',
    shuffle=False
)

# Расчёт веса классов (если есть дисбаланс)
class_counts = Counter(train_data.classes)
total = sum(class_counts.values())
class_weight = {i: total / (len(class_counts) * class_counts[i]) for i in class_counts}

# Базовая модель
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Замораживаем веса

# Надстройка
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

output_layer = Dense(1, activation='sigmoid') if class_mode == "binary" else Dense(num_classes, activation='softmax')
output = output_layer(x)

model = Model(inputs=base_model.input, outputs=output)


# Компиляция
loss = 'binary_crossentropy' if class_mode == "binary" else 'categorical_crossentropy'
model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])

# Ранняя остановка
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение
model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    class_weight=class_weight,
    callbacks=[early_stop]
)

# Оценка
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Точность модели на валидации: {test_accuracy:.2f}")

# Сохранение
model.save('image_classifier_improved.h5')
