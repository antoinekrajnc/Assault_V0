import tensorflow as tf

def preprocessing_image(img):
    img = tf.convert_to_tensor(img._force(), dtype=tf.float32)
    img = tf.transpose(img, [2, 0, 1])
    img = tf.reshape(img, [1] + [shape for shape in img.shape] + [1])
    return img