import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------
# 1. DEFINE DATASET PATHS
# -------------------------------------------
data_root = Path("C:/Valarr/Herbal Plant Classification/dataset")
train_dir = data_root / "train"
val_dir   = data_root / "validation"
test_dir  = data_root / "test"

# -------------------------------------------
# 2. CREATE DATAFRAMES
# -------------------------------------------
def create_df(directory: Path):
    filepaths = list(directory.glob("**/*.jpg"))
    labels = [fp.parent.name for fp in filepaths]
    df = pd.DataFrame({
        "Filepath": [str(fp) for fp in filepaths],  
        "Label": labels
    })
    return df.sample(frac=1).reset_index(drop=True)

train_df = create_df(train_dir)
val_df   = create_df(val_dir)
test_df  = create_df(test_dir)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))
print("\nClass distribution (Train):\n", train_df["Label"].value_counts())

# -------------------------------------------
# 3. IMAGE PREPROCESSING & AUGMENTATION
# -------------------------------------------
IMG_SIZE = 224  
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass indices:", train_generator.class_indices)

# -------------------------------------------
# 4. BUILD CNN MODEL
# -------------------------------------------
num_classes = len(train_generator.class_indices)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# -------------------------------------------
# 5. TRAIN THE MODEL
# -------------------------------------------
EPOCHS = 20

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

# -------------------------------------------
# 6. PLOT TRAINING HISTORY
# -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid()

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# -------------------------------------------
# 7. EVALUATE ON TEST SET
# -------------------------------------------
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

test_predictions = model.predict(test_generator)
test_pred_labels = np.argmax(test_predictions, axis=1)
true_labels = test_generator.classes

class_names = list(train_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(true_labels, test_pred_labels, target_names=class_names))

cm = confusion_matrix(true_labels, test_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# -------------------------------------------
# 8. SAVE THE MODEL
# -------------------------------------------
model.save('herbal_plant_classifier.h5')
print("\nModel saved as 'herbal_plant_classifier.h5'")


