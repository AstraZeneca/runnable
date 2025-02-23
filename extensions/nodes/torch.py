from typing import Any

from pydantic import Field

from runnable.nodes import CompositeNode


class TorchNode(CompositeNode):
    node_type: str = Field(default="torch", serialization_alias="type")
    num_gpus: int = Field(default=1)
    num_nodes: int = Field(default=1)
    is_composite: bool = True

    def get_summary(self) -> dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
            "num_gpus": self.num_gpus,
            "num_nodes": self.num_nodes,
        }

        return summary
