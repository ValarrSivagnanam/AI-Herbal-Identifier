import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
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
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

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
BATCH_SIZE = 16  

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
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
num_classes = len(train_generator.class_indices)

# -------------------------------------------
# 4. OPTION 1: IMPROVED CNN 
# -------------------------------------------
def build_improved_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# -------------------------------------------
# 5. OPTION 2: TRANSFER LEARNING 
# -------------------------------------------
def build_transfer_learning_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

USE_TRANSFER_LEARNING = True 

if USE_TRANSFER_LEARNING:
    model, base_model = build_transfer_learning_model()
    print("\nUsing Transfer Learning with MobileNetV2")
else:
    model = build_improved_cnn()
    print("\nUsing Improved CNN from scratch")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# -------------------------------------------
# 6. CALLBACKS FOR BETTER TRAINING
# -------------------------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    ModelCheckpoint(
        'best_herbal_classifier.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# -------------------------------------------
# 7. TRAIN THE MODEL
# -------------------------------------------
EPOCHS = 50  

print("\n" + "="*50)
print("TRAINING PHASE 1: Frozen base (Transfer Learning only)")
print("="*50)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------
# 8. FINE-TUNING (Transfer Learning only)
# -------------------------------------------
if USE_TRANSFER_LEARNING:
    print("\n" + "="*50)
    print("TRAINING PHASE 2: Fine-tuning last layers")
    print("="*50)
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )
    
    for key in history.history.keys():
        history.history[key].extend(history_fine.history[key])

# -------------------------------------------
# 9. PLOT TRAINING HISTORY
# -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------
# 10. EVALUATE ON TEST SET
# -------------------------------------------
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

test_predictions = model.predict(test_generator)
test_pred_labels = np.argmax(test_predictions, axis=1)
true_labels = test_generator.classes

class_names = list(train_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(true_labels, test_pred_labels, target_names=class_names))

cm = confusion_matrix(true_labels, test_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Herbal Plant Classification', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("\nPer-Class Accuracy:")
for class_name, acc in zip(class_names, per_class_accuracy):
    print(f"  {class_name:15s}: {acc:.2%}")

# -------------------------------------------
# 11. SAVE THE MODEL
# -------------------------------------------
model.save('herbal_plant_classifier_final.keras')
print("\n" + "="*50)
print("Model saved as 'herbal_plant_classifier_final.keras'")
print("Best model saved as 'best_herbal_classifier.keras'")
print("="*50)