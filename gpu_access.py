import tensorflow as tf

# Create TensorFlow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Define and compile your model
model = ...

# Train the model
model.fit(...)
