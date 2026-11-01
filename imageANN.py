
# MNIST Image Classification using Simple ANN


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Load Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize Data (0-255 → 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Build ANN Model
model = Sequential([
    Flatten(input_shape=(28, 28)),   # Converts 28x28 images to 784 vector
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes (digits 0–9)
])

# 4. Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train Model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# 6. Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_acc)

# 7. Predictions
predictions = model.predict(X_test[:5])

print("\nSample Predictions:")
for i in range(5):
    print("Predicted:", predictions[i].argmax(), " | Actual:", y_test[i])

# 8. Plot Training Curves
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 9. Save Model 
model.save("mnist_ann_model.keras")
print("\nModel saved as mnist_ann_model.keras")
