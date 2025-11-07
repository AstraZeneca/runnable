from typing import Any, Dict

from runnable import (  # pylint: disable=import-error  # pylint: disable=import-error
    exceptions,
    nodes,
)
from runnable.datastore import StepLog
from runnable.graph import Graph
from runnable.nodes import StepLog


class TestNode(nodes.BaseNode):
    """A concrete implementation of BaseNode for testing purposes."""

    def _get_on_failure_node(self) -> str:
        """Dummy implementation that returns empty string"""
        return ""

    def _get_next_node(self) -> str:
        """Dummy implementation that returns 'next'"""
        return "next"

    def _is_terminal_node(self) -> bool:
        """Dummy implementation that returns False"""
        return False

    def _get_catalog_settings(self) -> Dict[str, Any]:
        """Dummy implementation that returns empty dict"""
        return {}

    def _get_branch_by_name(self, branch_name: str) -> "Graph":
        """Dummy implementation that returns None"""
        return None

    def _get_executor_config(self, executor_type: str) -> str:
        """Dummy implementation that returns empty string"""
        return ""

    def _get_max_attempts(self) -> int:
        """Dummy implementation that returns 1"""
        return 1

    def execute(
        self, mock=False, map_variable=None, attempt_number: int = 1
    ) -> "StepLog":
        """Dummy implementation that returns None"""
        return None

    def execute_as_graph(self, map_variable=None):
        """Dummy implementation"""
        pass

    def fan_out(self, map_variable=None):
        """Dummy implementation"""
        pass

    def fan_in(self, map_variable=None):
        """Dummy implementation"""
        pass

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "TestNode":
        """Creates a TestNode from config dict"""
        return cls(**config)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a dummy summary"""
        return {
            "name": self.name,
            "type": self.node_type,
            "internal_name": self.internal_name,
        }



class TestTraversalNode(nodes.TraversalNode):
    """Concrete implementation of TraversalNode for testing"""

    def _get_catalog_settings(self) -> Dict[str, Any]:
        return {}

    def _get_branch_by_name(self, branch_name: str) -> Graph:
        return None

    def execute(
        self, mock=False, map_variable=None, attempt_number: int = 1
    ) -> StepLog:
        return StepLog(name=self.name, internal_name=self.internal_name)

    def execute_as_graph(self, map_variable=None):
        pass

    def fan_out(self, map_variable=None):
        pass

    def fan_in(self, map_variable=None):
        pass

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "TestTraversalNode":
        return cls(**config)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.node_type,
            "next": self.next_node,
            "on_failure": self.on_failure,
        }



class TestExecutableNode(nodes.ExecutableNode):
    """Concrete implementation of ExecutableNode for testing"""

    def _get_on_failure_node(self) -> str:
        return self.on_failure

    def _get_next_node(self) -> str:
        return self.next_node

    def _is_terminal_node(self) -> bool:
        return False

    def _get_catalog_settings(self) -> Dict[str, Any]:
        if self.catalog:
            return self.catalog.model_dump()
        return {}

    def _get_branch_by_name(self, branch_name: str) -> Graph:
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have branches"
        )

    def _get_executor_config(self, executor_type: str) -> str:
        return self.overrides.get(executor_type) or ""

    def _get_max_attempts(self) -> int:
        return self.max_attempts

    def execute(
        self, mock=False, map_variable=None, attempt_number: int = 1
    ) -> StepLog:
        step_log = StepLog(
            name=self.name,
            internal_name=self.internal_name,
            status="SUCCESS" if not mock else "MOCK",
        )
        return step_log

    def execute_as_graph(self, map_variable=None):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a graph"
        )

    def fan_in(self, map_variable=None):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a fan in"
        )

    def fan_out(self, map_variable=None):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a fan out"
        )

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "TestExecutableNode":
        return cls(**config)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.node_type,
            "next": self.next_node,
            "on_failure": self.on_failure,
            "max_attempts": self.max_attempts,
            "catalog": self.catalog.model_dump() if self.catalog else {},
        }



# Add these imports if not present
def test_base_node_initialization():
    """Test basic initialization of BaseNode through TestNode"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    assert node.name == "test_node"
    assert node.internal_name == "test.node"
    assert node.node_type == "test"


def test_base_node_parse_from_config():
    """Test creating node from config dictionary"""
    config = {
        "name": "test_node",
        "internal_name": "test.node",
        "node_type": "test",
    }

    node = TestNode.parse_from_config(config)
    assert node.name == "test_node"
    assert node.internal_name == "test.node"
    assert node.node_type == "test"
    assert node._get_next_node() == "next"  # TestNode implementation returns "next"


def test_base_node_get_summary():
    """Test getting node summary"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    summary = node.get_summary()
    assert summary["name"] == "test_node"
    assert summary["type"] == "test"
    assert summary["internal_name"] == "test.node"


def test_base_node_execute():
    """Test execute method returns None as per TestNode implementation"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    result = node.execute(mock=False)
    assert result is None


def test_base_node_execute_with_parameters():
    """Test execute method with various parameters"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    result = node.execute(mock=True, map_variable={"test": "value"}, attempt_number=2)
    assert result is None


def test_base_node_execute_as_graph():
    """Test execute_as_graph method"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    # Should not raise any exception as per TestNode implementation
    node.execute_as_graph(map_variable={"test": "value"})


def test_base_node_fan_operations():
    """Test fan_in and fan_out operations"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    # Should not raise any exceptions as per TestNode implementation
    node.fan_out(map_variable={"test": "value"})
    node.fan_in(map_variable={"test": "value"})


def test_base_node_get_branch_by_name():
    """Test getting branch by name"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    result = node._get_branch_by_name("test_branch")
    assert result is None  # As per TestNode implementation


def test_base_node_get_catalog_settings():
    """Test getting catalog settings"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    settings = node._get_catalog_settings()
    assert isinstance(settings, dict)
    assert len(settings) == 0  # As per TestNode implementation


def test_base_node_max_attempts():
    """Test getting max attempts"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    attempts = node._get_max_attempts()
    assert attempts == 1  # As per TestNode implementation


def test_base_node_terminal_status():
    """Test terminal node status"""
    node = TestNode(name="test_node", internal_name="test.node", node_type="test")

    is_terminal = node._is_terminal_node()
    assert is_terminal is False  # As per TestNode implementation


# Test implementations for each node type
def test_executable_node_overrides():
    """Test ExecutableNode with executor overrides"""
    node = TestExecutableNode(
        name="test_exec",
        internal_name="test.exec",
        node_type="executable",
        next_node="next_step",
        overrides={"local": "custom_config"},
    )

    assert node._get_executor_config("local") == "custom_config"
    assert node._get_executor_config("nonexistent") == ""


def test_executable_node_terminal_status():
    """Test ExecutableNode terminal status"""
    node = TestExecutableNode(
        name="test_exec",
        internal_name="test.exec",
        node_type="executable",
        next_node="next_step",
    )

    assert not node._is_terminal_node()


def test_executable_node_failure_handling():
    """Test ExecutableNode failure handling"""
    node = TestExecutableNode(
        name="test_exec",
        internal_name="test.exec",
        node_type="executable",
        next_node="next_step",
        on_failure="failure_node",
    )

    assert node._get_on_failure_node() == "failure_node"
    assert node._get_next_node() == "next_step"

    # Test neighbors includes both next and failure nodes
    neighbors = node._get_neighbors()
    assert len(neighbors) == 2
    assert "next_step" in neighbors
    assert "failure_node" in neighbors


def test_executable_node_parse_config():
    """Test ExecutableNode configuration parsing"""
    config = {
        "name": "test_exec",
        "internal_name": "test.exec",
        "node_type": "executable",
        "next_node": "next_step",
        "max_attempts": 3,
        "on_failure": "failure_node",
        "catalog": {"get": ["input1"], "put": ["output1"]},
        "overrides": {"local": "custom_config"},
    }

    node = TestExecutableNode.parse_from_config(config)

    assert node.name == "test_exec"
    assert node.internal_name == "test.exec"
    assert node.node_type == "executable"
    assert node.next_node == "next_step"
    assert node.max_attempts == 3
    assert node.on_failure == "failure_node"
    assert node._get_catalog_settings() == {
        "get": ["input1"],
        "put": ["output1"],
        "store_copy": True,
    }
    assert node._get_executor_config("local") == "custom_config"
