import tensorflow as tf
model = tf.keras.models.load_model(r"F:\Projects\Flower Classification\flower_classification_model_vgg_2.h5")
model.export("F:/Projects/Flower Classification/saved_model/")