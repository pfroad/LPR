import tensorflow as tf
from tensorflow.python.framework import graph_util
from google.protobuf import text_format

with tf.Session() as sess:
    latest_ckpt = tf.train.latest_checkpoint("./train_logs_50000")
    restore_saver = tf.train.import_meta_graph("./train_logs_50000/model.ckpt-29999.meta")

    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                 sess.graph_def,
                                                                 ["fc21/fc21", "fc22/fc22", "fc23/fc23", "fc24/fc24",
                                                                  "fc25/fc25", "fc26/fc26", "fc27/fc27"])
    # tf.train.write_graph(output_graph_def, "train_logs_50000", "pr1.pb", as_text=False)

    with tf.gfile.FastGFile("train_logs_50000\pr2.pbtxt", mode='w') as f:
        f.write(text_format.MessageToString(output_graph_def))
        # f.write(output_graph_def.SerializeToString())
