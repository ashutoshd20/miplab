from tensorflow.keras.datasets import cifar10

# Load and normalize the CIFAR10 dataset
(train_images, _), (test_images, _) = cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the batch size and the number of epochs
batch_size = 64
epochs = 10

import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.layers import Dense, Flatten, Reshape

# Let's define the transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        inputs = tf.reshape(inputs, (-1, 1, self.embed_dim))  # Reshape the inputs
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(tf.reshape(out1, (-1, self.embed_dim)))  # Flatten the output for FFN
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
class ImageTransformer(tf.keras.Model):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, ff_dim, num_transformer_blocks, **kwargs):
        super(ImageTransformer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.rescaling = Resizing(patch_size, patch_size)
        self.flatten = Flatten()
        self.dense1 = Dense(embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_blocks)]
        self.dense2 = Dense(patch_size*patch_size*3) # Assuming the image is in RGB
        self.reshape = Reshape((img_size, img_size, 3))

    def call(self, images):
        x = self.rescaling(images)
        x = self.flatten(x)
        x = self.dense1(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return x
image_transformer = ImageTransformer(
    img_size=32,
    patch_size=32,
    embed_dim=64,
    num_heads=2,
    ff_dim=32,
    num_transformer_blocks=2
)
image_transformer.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
image_transformer.fit(
    train_images,
    train_images,
    batch_size=1,
    epochs=epochs,
    validation_data=(test_images, test_images),
)

# Let's test it with one image from the test set
import matplotlib.pyplot as plt

# Predict the reconstruction of the first image in the test set
img = test_images[0]
reconstructed_img = image_transformer.predict(img[None, ...])[0]

# Plot the original and the reconstructed image
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[0].set_title('Original')
axes[1].imshow(reconstructed_img)
axes[1].set_title('Reconstructed')
plt.show()

