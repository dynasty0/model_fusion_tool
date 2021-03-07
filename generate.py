import call_node
from protos import ops_pb2
from google.protobuf import text_format
import tensorflow as tf
from utils import model_util
import sys

new_node_list = []

### 输入的配置文件路径
in_path = sys.argv[1]
### 输出的融合后的pb的路径
out_path = sys.argv[2]

ops_all = model_util.load_pbtxt_file(in_path)

for node in ops_all.node:
    a = call_node.CallNode(node)
    node_list, last_node = a.generate_node()
    new_node_list += node_list

graph_def = tf.compat.v1.GraphDef()
graph_def.node.extend(new_node_list)

if out_path[-3:] == 'txt':
    model_util._protobuf_to_file(graph_def, out_path)
elif out_path[-2:] == 'pb': 
    model_util.save_pb(graph_def, out_path)
