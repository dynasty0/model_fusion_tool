import cv2
import tensorflow as tf
import numpy  as np
from google.protobuf import text_format
from protos import ops_pb2

def load_graph_def(path):
    graph = tf.compat.v1.GraphDef()
    with open(path, 'rb') as f:
        graph.ParseFromString(f.read())
    return graph

def load_pbtxt_file(path):
    with open(path, 'r') as f:
        pbtxt_string = f.read()
        pbtxt = ops_pb2.NodeList()
        text_format.Merge(pbtxt_string, pbtxt)
    return pbtxt

def _protobuf_to_file(pb, path):
    with tf.io.gfile.GFile(path, 'w') as f:
        f.write(str(pb))

def save_pb(graph_def, pb_name):
    with tf.io.gfile.GFile(pb_name, "wb") as f:
        f.write(graph_def.SerializeToString())

def pbtxt2pb(pbtxt_path, pb_path):
    with open(pbtxt_path, 'r') as f:
        graph_def = tf.compat.v1.GraphDef()
        file_content = f.read()
        text_format.Merge(file_content, graph_def)
        save_pb(graph_def, pb_path)