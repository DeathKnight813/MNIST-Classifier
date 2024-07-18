import numpy as np
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

from matplotlib import pyplot as plt


#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# for i in range(9):
#     plt.subplot(330 + 1 + i)
#     plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
# plt.show()

# Load and preprocess the MNIST dataset
train_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0
train_y = to_categorical(train_y, 10)
test_y = to_categorical(test_y, 10)

# Build the MLP model
model = Sequential([
    Flatten(input_shape=(28, 28)),

    Dense(300, activation='relu'),
    Dense(100, activation='relu'),

    # Dense(128, activation='relu'),
    # Dense(64, activation='relu'),

    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_X, train_y, epochs=10, batch_size=32,validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_X, test_y)
print(f'Test accuracy: {test_acc}')


# Visualize some predictions
predictions = model.predict(test_X)
plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_X[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(predictions[i]))
plt.show()