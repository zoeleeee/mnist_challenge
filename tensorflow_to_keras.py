import tensorflow as tf
import keras
import sys
from model import Model

model_dir = sys.argv[-1]
checkpoint = tf.train.latest_checkpoint(model_dir)
model = Model()
saver = tf.train.Saver()

with tf.Session() as sess:
    # load weights for graph
    saver.restore(sess, checkpoint)

    # get all global variables (including model variables)
    vars_global = tf.global_variables()

    # get their name and value and put them into dictionary
    sess.as_default()
    model_vars = {}
    for var in vars_global:
        try:
            model_vars[var.name] = var.eval()
        except:
            print("For var={}, an exception occurred".format(var.name))

model = keras.Sequential([keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(28,28,32)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(20)
    ])

print(len(model.layers))
model.layers[0].set_weights(model_vars['Variable:0'])
model.layers[1].set_weights(model_vars['Variable_1:0'])
model.layers[2].set_weights(model_vars['Variable_2:0'])
model.layers[3].set_weights(model_vars['Variable_3:0'])
model.layers[4].set_weights(model_vars['Variable_4:0'])
model.layers[5].set_weights(model_vars['Variable_5:0'])
model.layers[6].set_weights(model_vars['Variable_6:0'])
model.layers[7].set_weights(model_vars['Variable_7:0'])

model.save(model_dir+'.h5')