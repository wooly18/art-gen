from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('Generator')
rescale = tf.keras.layers.Rescaling(127.5, offset=127.5)

for i in range(100):
    
    noise = tf.random.normal([1,100])
    sample = rescale(model(noise, training=False))

    plt.imshow(sample[0].numpy().astype("int16"))
    plt.axis("off")
    plt.savefig(f'outputs/{i}.png')