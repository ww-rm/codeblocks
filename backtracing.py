from typing import *


class BTNode:
    """"""

    @property
    def children(self) -> List["BTNode"]:
        return self.get_children()

    @property
    def children_count(self) -> int:
        return len(self.children)

    @property
    def isleaf(self) -> bool:
        return self.children_count <= 0

    @property
    def next_child(self) -> Union["BTNode", None]:
        if self.__next_child_idx >= self.children_count:
            return None
        child = self.children[self.__next_child_idx]
        self.__next_child_idx += 1
        return child

    def __init__(self) -> None:
        self.__next_child_idx = 0

    def get_children(self) -> List["BTNode"]:
        return []


class Backtracker:
    """"""

    def needprune(self, nodes: List[BTNode], next_node: BTNode) -> bool:
        """"""
        return False

    def find(self, nodes: List[BTNode]) -> None:
        """"""
        print(*nodes, sep=" -> ")

    def search(self, root: BTNode):
        """Search from root to leaf"""

        nodes = [root]
        while nodes:
            top = nodes[-1]

            # leaf node
            if top.isleaf:
                self.find(nodes)

            child = top.next_child

            # backward
            if child is None:
                nodes.pop()

            # forward
            else:
                if not self.needprune(nodes, child):
                    nodes.append(child)


class BinNode(BTNode):
    def __init__(self, level, value) -> None:
        super().__init__()
        self.level = level
        self.value = value

    def get_children(self) -> List["BTNode"]:
        if self.level < 4:
            return [BinNode(self.level + 1, 0), BinNode(self.level + 1, 1)]
        return []

    def __str__(self) -> str:
        return f"{self.value}"


class BinTracker(Backtracker):
    def needprune(self, nodes: List[BinNode], next_node: BinNode) -> bool:
        return sum(n.value for n in nodes) + next_node.value > 2


if __name__ == "__main__":
    BinTracker().search(BinNode(0, 1))
