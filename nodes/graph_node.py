from nodes.base_node import BaseNode
from utils import model_util
from utils import node_util
import tensorflow as tf

class GraphNode(BaseNode):
    def __init__(self, node):
        super().__init__(node)
        self.nnode = self.parse(self.node)
        self.graph = self.load_graph()

    def load_graph(self):
        path = self.nnode.path
        return model_util.load_graph_def(path)

    def modify(self):
        node_list = []
        prefix = self.nnode.prefix
        remove_node_list = self.nnode.node_to_remove
        modify_node_list = self.nnode.node_to_modify
        modify_info = self.nnode.node_to_modify_info
        for gnode in self.graph.node:
            if gnode.name in remove_node_list:
                continue
            info_id = 0
            if gnode.name in modify_node_list:
                for index, input in enumerate(gnode.input):
                    if info_id < len(modify_info) and input == modify_info[info_id].key:
                        gnode.input[index] = modify_info[info_id].value
                        info_id += 1
                        continue
                    gnode.input[index] = prefix + '/' + input
                gnode.name = prefix + '/' + gnode.name
                node_list.append(gnode)
            else:
                node_list.append(node_util.add_prefix(gnode, prefix))
        graph_def = tf.compat.v1.GraphDef()
        graph_def.node.extend(node_list)
        self.graph = graph_def

    def change_concat_input_ranges(self):
        node_util.change_concat_input_ranges_in_graph(self.graph.node)

    def graph_node_process(self):
        self.modify()
        self.change_concat_input_ranges()

    def generate_node(self):
        self.graph_node_process()
        return self.graph.node, self.graph.node[-1]