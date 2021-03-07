from nodes.base_node import BaseNode
from nodes import *
from protos import ops_pb2

node_dict = {
    ops_pb2.MyGraphDef: graph_node.GraphNode,
    ops_pb2.CustomOpDef: custom_node.CustomNode,
    ops_pb2.ConstDataDef: data_node.DataNode,
}

class CallNode(BaseNode):
    def __init__(self,node):
        super().__init__(node)

    def generate_node(self):
        nnode = self.parse(self.node)
        func = node_dict[type(nnode)]
        return func(self.node).generate_node()