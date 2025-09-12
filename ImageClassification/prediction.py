import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import load_model

loaded_model=load_model('baseline_model.h5')

(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train/255.0
X_test=X_test/255.0

X_train=X_train.reshape(X_train.shape[0],28*28)
X_test=X_test.reshape(X_test.shape[0],28*28)

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

sample_image=X_test[0].reshape(1,784)
predicted_probab=loaded_model.predict(sample_image)
predicted_class=predicted_probab.argmax()

print(f"predicted digit:{predicted_class}")

plt.imshow(sample_image.reshape(28,28),cmap='gray')
plt.title(f"Predicted DIgit:{predicted_class}")
plt.axis("off")
plt.show()
