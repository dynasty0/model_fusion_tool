from nodes.base_node import BaseNode
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import types_pb2
import tensorflow as tf 
from utils import node_util

type_dict = {
    "float": types_pb2.DT_FLOAT,
    "int": types_pb2.DT_INT32,
    "bool": types_pb2.DT_BOOL,
    "uint8": types_pb2.DT_UINT8,
}

class CustomNode(BaseNode):
    def __init__(self, node):
        super().__init__(node)
        self.nnode = self.parse(node)

    def parse_dims(self):
        dims = self.nnode.dims
        _output_shapes = self.nnode._output_shapes
        shape_list = []
        index_start = 0
        for dim in dims:
            shape = _output_shapes[index_start : index_start + dim]
            shape_list.append(tf.TensorShape(shape).as_proto())
            index_start += dim
        return shape_list
    
    def parse_types(self):
        _output_types = self.nnode._output_types
        type_list = []
        for types in _output_types:
            type_list.append(type_dict[types])
        return type_list

    def add_fake_quant_nodes(self):
        node_list = []
        for id, node_name, min, max in zip(
            self.nnode.fake_quant_ids, self.nnode.fake_quant_names, self.nnode.fake_quant_mins, self.nnode.fake_quant_maxs
        ):
            custom_output_node_name = self.nnode.name + ":" + str(id)
            _node_list, _fake_quant_node = node_util.make_fake_quant_node_set(node_name, min, max, custom_output_node_name)
            node_list += _node_list
        return node_list

    def add_identity_node(self):
        node_list = []
        for id, node_name, dtype in zip(self.nnode.identity_ids, self.nnode.identity_names, self.nnode.identity_types):
            if id in self.nnode.fake_quant_ids:
                fake_quant_ids = list(self.nnode.fake_quant_ids)
                input_node_name = self.nnode.fake_quant_names[fake_quant_ids.index(id)]
            else:
                input_node_name = self.nnode.name + ":" + str(id)
            identity_node = node_util.make_identity_node(node_name, type_dict[dtype], input_node_name)
            node_list.append(identity_node)
        return node_list

    def make_custom_node(self):
        node = node_def_pb2.NodeDef()
        node.name = self.nnode.name
        input_list = self.nnode.inputs
        node.op = self.nnode.type
        node.input.extend(input_list)

        node.attr['_output_types'].list.type.extend(self.parse_types())
        node.attr['_output_shapes'].list.shape.extend(self.parse_dims())
        node.attr['_output_quantized'].b = self.nnode._output_quantized
        node.attr['_support_output_type_float_in_quantized_op'].b = self.nnode._support_output_type_float_in_quantized_op

        for attr_name in self.nnode.attr:
            value_org = self.nnode.attr[attr_name]
            value_type = value_org.WhichOneof("value")
            value = getattr(value_org, value_type)
            if value_type == 'f':
                node.attr[attr_name].f = value
            elif value_type == 'i':
                node.attr[attr_name].i = value
            elif value_type == 'b':
                node.attr[attr_name].b = value
            else:
                raise NotImplementedError("目前支持类型为：float， int, bool的attr")
        return node

    def generate_node(self):
        node_list = []
        node = self.make_custom_node()
        node_list.append(node)
        node_list += self.add_fake_quant_nodes()
        node_list += self.add_identity_node()
        last_node = node_list[-1]
        return node_list, last_node