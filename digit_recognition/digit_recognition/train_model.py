import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data to add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")

# Build the Neural Network Model
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(28, 28, 1)),
    
    # Convolutional layers
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 classes for digits 0-9
])

# Print model summary
model.summary()

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
print("\nEvaluating on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# Save the model
model.save("digit_recognition_model.h5")
print("\nModel saved as 'digit_recognition_model.h5'")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history plot saved as 'training_history.png'")

# Test on some random images
n_samples = 10
random_indices = np.random.randint(0, len(x_test), n_samples)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(random_indices):
    img = x_test[idx]
    true_label = y_test[idx]
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, 0), verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    plt.subplot(1, n_samples, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"True: {true_label}\nPred: {predicted_label}\n({confidence:.1f}%)")
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png')
print("Sample predictions saved as 'sample_predictions.png'")
print("\nTraining complete!")