import tensorflow as tf

model = tf.keras.models.load_model(
    "bone_fracture_model.h5",
    compile=False,
    custom_objects={}
)

print("MODEL LOADED")