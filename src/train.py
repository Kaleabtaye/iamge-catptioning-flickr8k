import tensorflow as tf
from data_loader import prepare_dataset
from model import ImageCaptioningModel

# Load pre-extracted features and captions
feature_path = '../data/image_features.pkl'
caption_path = '../data/Flickr8k_text/captions.txt'

features, captions, tokenizer = prepare_dataset(feature_path, caption_path)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
units = 256
batch_size = 64
epochs = 10

dataset = tf.data.Dataset.from_tensor_slices((features, captions))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

model = ImageCaptioningModel(vocab_size, embedding_dim, units)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.not_equal(real, 0)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(img_tensor, target):
    loss = 0
    decoder_input = target[:, :-1]
    real_output = target[:, 1:]

    with tf.GradientTape() as tape:
        predictions = model([img_tensor, decoder_input])
        loss = loss_function(real_output, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for (batch, (image_features, target)) in enumerate(dataset):
        batch_loss = train_step(image_features, target)
        total_loss += batch_loss

        if batch % 10 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')

    print(f'Epoch {epoch + 1} Loss {total_loss / (batch + 1):.4f}')

# Save model weights
model.save_weights('./saved_models/image_captioning_model.h5')