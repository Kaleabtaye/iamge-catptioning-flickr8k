import tensorflow as tf
from tensorflow.keras import layers

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(ImageCaptioningModel, self).__init__()
        # self.encoder = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')
        self.image_dense = layers.Dense(units, activation='relu')
        self.image_dropout = layers.Dropout(0.2)

        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        self.lstm_dropout = layers.Dropout(0.2)
        self.fc = layers.Dense(vocab_size)


    def call(self, inputs):
        image_features, captions = inputs

        # Encode the images
        # features = self.encoder(images)
        features = self.image_dense(image_features)
        features = self.image_dropout(features)
        features = tf.expand_dims(features, 1)

        # Embed the captions
        captions = self.embedding(captions)

        # Concatenate image features and caption embeddings
        inputs = tf.concat([features, captions], axis=1)

        lstm_output, _, _ = self.lstm(inputs)

        outputs = self.fc(lstm_output)
        return outputs