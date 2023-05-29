import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import numpy as np

# Load dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reduce the training data to 10% while maintaining the class distribution
train_images, _, train_labels, _ = train_test_split(train_images, train_labels, train_size=0.1, stratify=train_labels,
                                                    random_state=42)

# Reduce the test data to 10% while maintaining the class distribution
test_images, _, test_labels, _ = train_test_split(test_images, test_labels, train_size=0.1, stratify=test_labels,
                                                  random_state=42)

# Split the remaining training data into training and validation data (80:20)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
                                                                      stratify=train_labels, random_state=42)

print("Number of images in train_images:", train_images.shape[0])
print("Number of images in train_labels:", train_labels.shape[0])
print("Number of images in val_images:", val_images.shape[0])
print("Number of images in val_labels:", val_labels.shape[0])
print("Number of images in test_images:", test_images.shape[0])
print("Number of images in test_labels:", test_labels.shape[0])


def create_model():
    num_classes = 10
    input_shape = (32, 32, 3)

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

model = create_model()

trainable_weights = []
for layer in model.layers:
    layer_param_count = layer.count_params()
    if layer_param_count != 0:
        trainable_weights.append(layer_param_count)
#flat_weights = [[1,4],[2,1,1]]
#flat_weights = [1,2,3,4,5,6]
size=trainable_weights
flat_weights = [np.random.uniform(size=size[1])]
#flat_weights=trainable_weights


def reshape_weights(flat_weights):
    flat_weights = np.array(flat_weights)  # Add this line to convert the input to a Numpy array
    shapes = [w.shape for w in model.get_weights()]
    print(shapes)
    weights = []
    index = 0
    for shape in shapes:
        size = np.prod(shape)
        weights.append(flat_weights[index:index + size].reshape(shape))
        index += size
    return weights

print(reshape_weights(flat_weights))

