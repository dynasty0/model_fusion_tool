from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tensor_shape

import tensorflow as tf
import numpy as np

type_dict = {
    np.dtype(np.int32): types_pb2.DT_INT32,
    np.dtype(np.float32): types_pb2.DT_FLOAT,
    np.dtype(np.float64): types_pb2.DT_FLOAT,
    np.dtype(np.uint8): types_pb2.DT_UINT8,
    np.dtype(np.bool_): types_pb2.DT_BOOL,
}

def parse(node):
    one_of_type = node.WhichOneof("op")
    return getattr(node, one_of_type)

def make_const_node(data, node_name):
    data_type = type_dict[data.dtype]
    data_shape = data.shape
    node = node_def_pb2.NodeDef()
    node.name = node_name
    node.op = 'Const'
    node.attr['dtype'].type = data_type
    tensor_proto = tensor_pb2.TensorProto(
        dtype = data_type,
        tensor_shape = tensor_shape.as_shape(data_shape).as_proto()
    ) 
    tensor_proto.tensor_content = data.tostring()
    node.attr['value'].tensor.CopyFrom(tensor_proto)

    return node

def make_const_val_node(data, node_name):
    data_type = types_pb2.DT_FLOAT
    node = node_def_pb2.NodeDef()
    node.name = node_name
    node.op = 'Const'
    node.attr['dtype'].type = data_type
    tensor_proto = tensor_util.make_tensor_proto(data, dtype = data_type)
    node.attr['value'].tensor.CopyFrom(tensor_proto)
    return node

def make_placeholder_node(node_name, data_type, shape=[-1,320,320,3]):
    node = node_def_pb2.NodeDef()
    node.name = node_name
    node.op = 'Placeholder'
    node.attr['dtype'].type = data_type
    tensor_shape = tensor_shape_pb2.TensorShapeProto()
    for _shape in shape:
        tensor_shape.dim.add().size = _shape
    node.attr["shape"].shape.CopyFrom(tensor_shape)
    return node

def make_read_node(input_node_name, data_type):
    node = node_def_pb2.NodeDef()
    node.name = input_node_name + '/read'
    node.op = 'Identity'
    node.input.extend([input_node_name])
    node.attr['T'].type = data_type
    node.attr['_class'].list.s.extend([("loc:@"+input_node_name).encode()])
    return node

def make_identity_node(node_name, data_type, input_node_name):
    node = node_def_pb2.NodeDef()
    node.name = node_name
    node.op = 'Identity'
    node.input.extend([input_node_name])
    node.attr['T'].type = data_type
    return node

def make_fakeQuant_node(node_name, input_list):
    node = node_def_pb2.NodeDef()
    node.name = node_name
    node.op = "FakeQuantWithMinMaxVars"
    node.input.extend(input_list)
    node.attr["narrow_range"].b = False
    node.attr["num_bits"].i = 8
    return node

def make_cropResize_node(node_name, input_list, output_shape=[3,64,64,3]):
    node = node_def_pb2.NodeDef()
    node.name = node_name
    node.op = "CropAndResize"
    node.input.extend(input_list)
    node.attr['crop_height'].i = output_shape[1]
    node.attr['crop_width'].i = output_shape[2]
    node.attr['align_corners'].b = True
    node.attr['method'].i = 0
    mode.attr['_output_types'].list.type.extend([types_pb2.DT_FLOAT,])
    node.attr['_output_shapes'].list.shape.extend([tf.TensorShape(output_shape).as_proto(),])
    node.attr['_output_quantized'].b = True
    return node

def make_cropResize_node_v2(node_name, input_list, output_shape=[3,64,64,3]):
    node = node_def_pb2.NodeDef()
    node.name = node_name
    node.op = "CropAndResize"
    node.input.extend(input_list)
    node.attr['crop_height'].i = output_shape[1]
    node.attr['crop_width'].i = output_shape[2]
    node.attr['align_corners'].b = True
    node.attr['method'].i = 0
    node.attr['_output_types'].list.type.extend([types_pb2.DT_FLOAT,types_pb2.DT_FLOAT,types_pb2.DT_FLOAT])
    node.attr['_output_shapes'].list.shape.extend([tf.TensorShape(output_shape).as_proto(),tf.TensorShape(output_shape[0]).as_proto(),tf.TensorShape(output_shape[3]).as_proto()])
    node.attr['_output_quantized'].b = True
    return node

def add_prefix(node, prefix):
    if node.name[-4:] == 'read':
        for index, value in enumerate(node.attr['_class'].list.s):
            node.attr['_class'].list.s[index] = ("loc:@" + prefix + '/' + value.decode()[5:]).encode()
    node.name = prefix + '/' + node.name
    for index, input in enumerate(node.input):
        node.input[index] = prefix + '/' + input
    return node

def make_fake_quant_node_set(node_name, min, max, input_node_name):
    node_list = []
    fake_quant_node_min = make_const_val_node(min, node_name + '/min')
    fake_quant_node_min_read = make_read_node(fake_quant_node_min.name, types_pb2.DT_FLOAT)
    
    node_list.append(fake_quant_node_min)
    node_list.append(fake_quant_node_min_read)

    fake_quant_node_max = make_const_val_node(max, node_name + '/max')
    fake_quant_node_max_read = make_read_node(fake_quant_node_max.name, types_pb2.DT_FLOAT)
    
    node_list.append(fake_quant_node_max)
    node_list.append(fake_quant_node_max_read)

    fake_quant_node = make_fakeQuant_node(node_name, [input_node_name, fake_quant_node_min_read.name, fake_quant_node_max_read.name])
    node_list.append(fake_quant_node)
    return node_list, fake_quant_node

def find_node(graph_node_list, input_node_name):
    for node in graph_node_list:
        if node.name == input_node_name:
            return node

def find_concat_node(graph_node_list):
    node_list = []
    for node in graph_node_list:
        if node.op == 'ConcatV2':
            node_list.append(node)
    return node_list

def find_min_max(graph_node_list, input_node_name):
    node_return = []
    node = find_node(graph_node_list, input_node_name)
    if node:
        for index in range(0, len(node.input) - 1):
            if node.input[index].endswith('FakeQuantWithMinMaxVars'):
                node_fake_quant = find_node(graph_node_list, node.input[index])
                val_node = []
                for index2 in range(1, len(node_fake_quant.input)):
                    read_node = find_node(graph_node_list, node_fake_quant.input[index2])
                    val_node.append(find_node(graph_node_list, read_node.input[0]))
                node_return.append(val_node)
            else:
                node_name_to_find = node.input[index]
                while node_name_to_find.endswith('FakeQuantWithMinMaxVars') == False:
                    node_to_find = find_node(graph_node_list, node_name_to_find)
                    node_name_to_find = node_to_find.input[0]
                node_fake_quant = find_node(graph_node_list, node_to_find.input[0])
                val_node = []
                for index2 in range(1, len(node_fake_quant.input)):
                    read_node = find_node(graph_node_list, node_fake_quant.input[index2])
                    val_node.append(find_node(graph_node_list, read_node.input[0]))
                node_return.append(val_node)
    return node_return

def modify_min_max_in_concat(graph_node_list, input_node_name):
    res = find_min_max(graph_node_list, input_node_name)
    if res is None:
        return 
    min = res[0][0].attr['value'].tensor.float_val[0]
    max = res[0][1].attr['value'].tensor.float_val[0]
    for i in range(1, len(res)):
        if min > res[i][0].attr['value'].tensor.float_val[0]:
            min = res[i][0].attr['value'].tensor.float_val[0]
        if max < res[i][1].attr['value'].tensor.float_val[0]:
            max = res[i][1].attr['value'].tensor.float_val[0]
    for i in range(0, len(res)):
        res[i][0].attr['value'].tensor.float_val[0] = min
        res[i][1].attr['value'].tensor.float_val[0] = max

def change_concat_input_ranges_in_graph(graph_node_list):
    node_list = find_concat_node(graph_node_list)
    for node in node_list:
        modify_min_max_in_concat(graph_node_list, node.name)