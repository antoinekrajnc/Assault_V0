import tensorflow as tf

def preprocessing_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
    img = img /255.0
    return img