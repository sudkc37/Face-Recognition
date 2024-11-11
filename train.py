import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.utils import Progbar
import os


def embedding():
    inputs = Input(shape=(105, 105, 3), name='input_image')
    layer1 = Conv2D(64, (10, 10), activation='relu')(inputs)
    pool1 = MaxPooling2D(64, (2, 2), padding='same')(layer1)

    layer2 = Conv2D(128, (7, 7), activation='relu')(pool1)
    pool2 = MaxPooling2D(64, (2, 2), padding='same')(layer2 )

    layer3 = Conv2D(128, (4, 4), activation='relu')(pool2)
    pool3 = MaxPooling2D(64, (2, 2), padding='same')(layer3)

    layer4 = Conv2D(256, (4, 4), activation='relu')(pool3)
    emb = Flatten()(layer4)  # Flattening into a single dimension
    dense = Dense(4096, activation='sigmoid')(emb)
    return Model(inputs=inputs, outputs=dense, name='embedding_model')

embedding_model = embedding()

# Define the custom Siamese distance layer
class siameseL1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Define the Siamese network
def siamese_model():
    input_image = Input(name='input_image', shape=(105, 105, 3))  # Anchor image
    validation_image = Input(name='validation_image', shape=(105, 105, 3))  # Positive or negative image

    embedding_anchor = embedding_model(input_image)
    embedding_validation = embedding_model(validation_image)

    siamese_layer = siameseL1Dist(name='distance')
    distance = siamese_layer(embedding_anchor, embedding_validation)

    # Classification layer for final output
    output = Dense(1, activation='sigmoid')(distance)
    return Model(inputs=[input_image, validation_image], outputs=output, name='SiameseNetwork')

siamese_model = siamese_model()

# Binary cross-entropy loss and optimizer setup
binary_crossentropy = tf.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4)

# Establish checkpoints
checkpoint_directory = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, siamese_model=siamese_model)

# Training step function
@tf.function
def train_steps(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]  # Get anchor and positive/negative images
        y = batch[2]   # Labels

        # Forward pass and loss
        yhat = siamese_model(X, training=True)
       
        loss = binary_crossentropy(y, yhat)
    print(loss)

    # Calculate gradients
    gradients = tape.gradient(loss, siamese_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, siamese_model.trainable_variables))

    return loss

# Training loop
def training_loop(data, epochs, steps_per_epoch):
    for epoch in range(1, epochs + 1):
        print('\n Epoch {}/{}'.format(epoch, epochs))
        bar = Progbar(steps_per_epoch)

        for idx, batch in enumerate(data):
            train_steps(batch)
            bar.update(idx+1)
            
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
