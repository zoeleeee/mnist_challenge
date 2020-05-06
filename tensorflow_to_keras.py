import tensorflow as tf
import keras
import sys

model_dir = sys.argv[-1]
checkpoint = tf.train.latest_checkpoint(model_dir)
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
    print(model_vars.keys())