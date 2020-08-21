#content = load_img('content.jpg').astype('uint8')
import tensorflow as tf
from PIL import Image
from keras import *
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


#Here are the content and style images we will use:
plt.figure(figsize=(10,10))


content = load_img('/root/Desktop/StyleTransfer/contents.jpg')
style = load_img('/root/Desktop/StyleTransfer/styles.jpg')

plt.subplot(1,2,1)
plt.imshow(content)
plt.subplot(1,2,2)
plt.imshow(style)
plt.show()

