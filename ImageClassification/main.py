#Step 1 : Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense

(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train/255.0
X_test=X_test/255.0
"""
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Unique labels:", np.unique(y_train))
"""

"""
plt.figure(figsize=(10,4))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.show()
"""
X_train=X_train.reshape(X_train.shape[0],28*28)
X_test=X_test.reshape(X_test.shape[0],28*28)

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


"""
print("X_train shape after preprocessing:", X_train.shape)
print("X_test shape after preprocessing:", X_test.shape)
print("y_train shape after preprocessing:", y_train.shape)
print("y_test shape after preprocessing:", y_test.shape)
"""

model=Sequential([
    Dense(128,activation="relu",input_shape=(784,)),
    Dense(10,activation="softmax")
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history=model.fit(X_train,y_train,epochs=10,batch_size=42,validation_data=(X_test,y_test))


model.save("baseline_model.h5")
"""plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label="Train Accuracy")
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'],label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss Over Epochs")
plt.legend()

plt.show()"""