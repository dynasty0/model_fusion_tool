from nodes.base_node import BaseNode
import sys

from utils import node_util

class DataNode(BaseNode):
    def __init__(self, node):
        super().__init__(node)

    def make_data(self,func_path, func_name):
        if func_path.startswith('..'):
            sys.path.append("..")
            module_path = func_path[3:]
        elif func_path.startswith('.'):
            module_path = func_path[2:]
        else:
            module_path = func_path
        module_path = module_path.replace('/', '.')[:-3]
        exec_str = "from " + module_path + " import *"
        exec(exec_str)
        return eval(func_name)()

    def data_node_process(self, node):
        node_list = []
        nnode = node
        data = self.make_data(nnode.func_path, nnode.func_name)
        data_node = node_util.make_const_node(data, nnode.name + '/data')
        read_node = node_util.make_read_node(data_node.name, node_util.type_dict[data.dtype])
        node_list.append(data_node)
        node_list.append(read_node)
        if nnode.add_fake_quant_flag:
            data_fake_quant_node_list, data_fake_quant_node = \
                node_util.make_fake_quant_node_set(nnode.name + '/fake_quant', nnode.min[0], nnode.max[0], read_node.name)
            node_list = node_list + data_fake_quant_node_list
            return node_list, data_fake_quant_node
        else:
            return node_list, read_node

    def generate_node(self):
        nnode = self.parse(self.node)
        return self.data_node_process(nnode)    