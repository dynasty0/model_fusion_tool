from abc import ABC, abstractmethod
from protos import ops_pb2

class BaseNode(ABC):
    def __init__(self, node):
        self.node = node

    def parse(self, node):
        one_of_type = node.WhichOneof('op')
        return getattr(node, one_of_type)

    @abstractmethod
    def generate_node(self):
        pass