from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# DATA
#loads data into 4 numpy arrays
(train_images , train_labels), (test_images, test_labels) = mnist.load_data()

# layers built of 2 Dense layers (fully-connected/densely conected)
# the second layer is 10-way softmax layer, it will return a probability score for 10 digits 0-9
network = models.Sequential()
network.add(layers.Dense(512,activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

#COMPILATION
# need a loss function - difference between prediction and verified output
# need an optimizer - mechanism when network will update itself based on data and loss function
# need metwrics to monitor success/failure
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#PREPROCESSING
#data needs to be in format that the NN expects as input
#before, it was uint8 array of shape (60000, 28, 28) but now we make it a float32 array with [0-1) float values
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255
#prepare the labels to categorical format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#TRAIN
print("\n\nStart Training")
network.fit(train_images, train_labels, epochs=5, batch_size=128)
print("Done training.\n\n")

#TEST
print("\n\nStart Testing")
testloss, testacc = network.evaluate(test_images, test_labels)
print("test accuracy: " + str(testacc))





