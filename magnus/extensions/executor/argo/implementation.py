import json
import logging
import random
import shlex
import string
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, field_serializer
from pydantic.functional_serializers import PlainSerializer
from ruamel.yaml import YAML
from typing_extensions import Annotated

from magnus import defaults, exceptions, integration, utils
from magnus.extensions.executor import GenericExecutor
from magnus.extensions.nodes import DagNode, MapNode, ParallelNode
from magnus.graph import Graph, create_node, search_node_by_internal_name
from magnus.nodes import BaseNode

logger = logging.getLogger(defaults.NAME)


class SecretEnvVar(BaseModel):
    """
    Renders:
    env:
      - name: MYSECRETPASSWORD
        valueFrom:
          secretKeyRef:
            name: my-secret
            key: mypassword
    """

    environment_variable: str = Field(serialization_alias="name")
    secret_name: str = Field(exclude=True)
    secret_key: str = Field(exclude=True)

    @computed_field  # type: ignore
    @property
    def valueFrom(self) -> Dict[str, Dict[str, str]]:
        return {
            "secretKeyRef": {
                "name": self.secret_name,
                "key": self.secret_key,
            }
        }


class EnvVar(BaseModel):
    """
    Renders:
    parameters: # in arguments
      - name: x
        value: 3 # This is optional for workflow parameters

    """

    name: str
    value: Union[str, int, float] = ""


class Request(BaseModel):
    """
    The default requests
    """

    memory: str = "1Gi"
    cpu: str = "250m"


class Retry(BaseModel):
    limit: int = 0
    retry_policy: str = Field(default="Always", serialization_alias="retryPolicy")

    @field_serializer("limit")
    def cast_limit_as_str(self, limit: int, _info) -> str:
        return str(limit)


VendorGPU = Annotated[
    Optional[int],
    PlainSerializer(lambda x: str(x), return_type=str, when_used="unless-none"),
]


class Limit(Request):
    """
    The default limits
    """

    gpu: VendorGPU = Field(default=None, serialization_alias="nvidia.com/gpu")


class Toleration(BaseModel):
    effect: str
    key: str
    operator: str
    value: str


class UserControls(BaseModel):
    """
    These are all the controls given to the user at the config level per step or globally.
    """

    limits: Limit = Field(default=Limit(), serialization_alias="limits")
    requests: Request = Field(default=Request(), serialization_alias="requests")
    image_pull_policy: str = Field(default="", serialization_alias="imagePullPolicy")
    # Better to leave it empty, as it is a sensible default
    # https://argoproj.github.io/argo-workflows/fields/#container
    node_selector: dict = Field(default={}, serialization_alias="nodeSelector")
    tolerations: Optional[List[Toleration]] = None
    # Note: retry seems to mess with the graph viz but works well with gant chart.
    retry_strategy: Retry = Field(default=Retry(), serialization_alias="retryStrategy")
    active_deadline_seconds: int = Field(default=60 * 60 * 2, serialization_alias="activeDeadlineSeconds", gt=0)
    # Note: Volume mounts cannot be defined here

    # Should not be populated by user!!
    _env_vars: List[EnvVar] = PrivateAttr(default=[])
    _secrets_from_k8s: List[SecretEnvVar] = PrivateAttr(default=[])

    @computed_field  # type: ignore
    @property
    def env(self) -> Optional[List[Union[EnvVar, SecretEnvVar]]]:
        if not self._env_vars and not self._secrets_from_k8s:
            return None

        return self._env_vars + self._secrets_from_k8s


ShlexCommand = Annotated[str, PlainSerializer(lambda x: shlex.split(x), return_type=List[str])]


class DefaultContainer(BaseModel):
    """
    The non-defaulted config + these variables make per step configuration.
    """

    image: str
    command: ShlexCommand
    volume_mounts: Optional[List["ContainerVolume"]] = Field(default=None, serialization_alias="volumeMounts")

    _env_vars: List[EnvVar] = PrivateAttr(default=[])
    _secrets_from_k8s: List[SecretEnvVar] = PrivateAttr(default=[])

    @computed_field  # type: ignore
    @property
    def env(self) -> Optional[List[Union[EnvVar, SecretEnvVar]]]:
        if not self._env_vars and not self._secrets_from_k8s:
            return None

        return self._env_vars + self._secrets_from_k8s


class Container(DefaultContainer, UserControls):
    ...


class Parameter(BaseModel):
    name: str
    value: Optional[str] = None


class OutputParameter(Parameter):
    """
    Renders:
    - name: step-name
      valueFrom:
        path: /tmp/output.txt
    """

    path: str = Field(default="/tmp/output.txt", exclude=True)

    @computed_field  # type: ignore
    @property
    def valueFrom(self) -> Dict[str, str]:
        return {"path": self.path}


class Argument(BaseModel):
    """
    Templates are called with arguments, which become inputs for the template
    Renders:
    arguments:
      parameters:
        - name: The name of the parameter
          value: The value of the parameter
    """

    name: str
    value: str


class TaskTemplate(BaseModel):
    """
    dag:
        tasks:
          name: A
            template: nested-diamond
            arguments:
                parameters: [{name: message, value: A}]
    """

    name: str
    template: str
    depends: List[str] = []
    arguments: Optional[List[Argument]] = Field(default=None)
    with_param: Optional[str] = Field(default=None, serialization_alias="withParam")

    @field_serializer("depends")
    def transform_depends_as_str(self, depends: List[str]) -> str:
        return " || ".join(depends)

    @field_serializer("arguments", when_used="unless-none")
    def empty_arguments_to_none(self, arguments: List[Argument]) -> Dict[str, List[Argument]]:
        return {"parameters": arguments}


class ContainerTemplate(BaseModel):
    # These templates are used for actual execution nodes.
    name: str
    container: Union[Container, DefaultContainer]
    outputs: Optional[List[OutputParameter]] = Field(default=None, serialization_alias="outputs")
    inputs: Optional[List[Parameter]] = Field(default=None, serialization_alias="inputs")

    def __hash__(self):
        return hash(self.name)

    @field_serializer("outputs", when_used="unless-none")
    def empty_outputs_to_none(self, outputs: List[OutputParameter]) -> Dict[str, List[OutputParameter]]:
        return {"parameters": outputs}

    @field_serializer("inputs", when_used="unless-none")
    def empty_inputs_to_none(self, inputs: List[Parameter]) -> Dict[str, List[Parameter]]:
        return {"parameters": inputs}


class DagTemplate(BaseModel):
    name: str = "magnus-dag"
    tasks: List[TaskTemplate] = Field(default=[], exclude=True)
    inputs: Optional[List[Parameter]] = Field(default=None, serialization_alias="inputs")
    parallelism: Optional[int] = None
    fail_fast: bool = Field(default=True, serialization_alias="failFast")

    @computed_field  # type: ignore
    @property
    def dag(self) -> Dict[str, List[TaskTemplate]]:
        return {"tasks": self.tasks}

    @field_serializer("inputs", when_used="unless-none")
    def reshape_inputs(self, inputs: List[Parameter], _info) -> Dict[str, List[Parameter]]:
        return {"parameters": inputs}


class Volume(BaseModel):
    """
    spec config requires, name and persistentVolumeClaim
    step requires name and mountPath
    """

    name: str
    claim: str = Field(exclude=True)
    mount_path: str = Field(serialization_alias="mountPath", exclude=True)

    @computed_field  # type: ignore
    @property
    def persistentVolumeClaim(self) -> Dict[str, str]:
        return {"claimName": self.claim}


class ContainerVolume(BaseModel):
    name: str
    mount_path: str = Field(serialization_alias="mountPath")


class UserVolumeMounts(BaseModel):
    """
    The volume specification as user defines it.
    """

    name: str  # This is the name of the PVC on K8s
    mount_path: str  # This is mount path on the container


class Spec(BaseModel):
    entrypoint: str = "magnus-dag"
    templates: List[Union[DagTemplate, ContainerTemplate]] = Field(default=[])
    service_account_name: Optional[str] = Field(default=None, serialization_alias="serviceAccountName")
    arguments: Optional[List[EnvVar]] = Field(default=None)
    persistent_volumes: List[UserVolumeMounts] = Field(default=[], exclude=True)
    template_defaults: Optional[UserControls] = Field(default=None, serialization_alias="templateDefaults")
    parallelism: Optional[int] = None

    @computed_field  # type: ignore
    @property
    def volumes(self) -> List[Volume]:
        volumes: List[Volume] = []
        claim_names = {}
        for i, user_volume in enumerate(self.persistent_volumes):
            if user_volume.name in claim_names:
                raise Exception(f"Duplicate claim name {user_volume.name}")
            claim_names[user_volume.name] = user_volume.name

            volume = Volume(name=f"executor-{i}", claim=user_volume.name, mount_path=user_volume.mount_path)
            volumes.append(volume)
        return volumes

    @field_serializer("arguments", when_used="unless-none")
    def reshape_arguments(self, arguments: List[EnvVar], _info) -> Dict[str, List[EnvVar]]:
        return {"parameters": arguments}


class Workflow(BaseModel):
    api_version: str = Field(default="argoproj.io/v1alpha1", serialization_alias="apiVersion")
    kind: str = "Workflow"
    metadata: dict = {"generateName": "magnus-dag-"}
    spec: Spec


class NodeRenderer(ABC):
    allowed_node_types: List[str] = []

    def __init__(self, executor: "ArgoExecutor", node: BaseNode) -> None:
        self.executor = executor
        self.node = node

    @abstractmethod
    def render(self, list_of_iter_values: Optional[List] = None):
        pass


class ExecutionNode(NodeRenderer):
    allowed_node_types = ["task", "stub", "success", "fail"]

    def render(self, list_of_iter_values: Optional[List] = None):
        """
        Compose the map variable and create the execution command.
        Create an input to the command.
        create_container_template : creates an argument for the list of iter values
        """
        map_variable = self.executor.compose_map_variable(list_of_iter_values)
        command = utils.get_node_execution_command(
            self.node,
            over_write_run_id=self.executor.run_id_placeholder,
            map_variable=map_variable,
        )

        inputs = []
        if list_of_iter_values:
            for val in list_of_iter_values:
                inputs.append(Parameter(name=val))

        # Create the container template
        container_template = self.executor.create_container_template(
            working_on=self.node,
            command=command,
            inputs=inputs,
        )

        self.executor._container_templates.append(container_template)


class DagNodeRenderer(NodeRenderer):
    allowed_node_types = ["dag"]

    def render(self, list_of_iter_values: Optional[List] = None):
        self.node = cast(DagNode, self.node)
        task_template_arguments = []
        dag_inputs = []
        if list_of_iter_values:
            for value in list_of_iter_values:
                task_template_arguments.append(Argument(name=value, value="{{inputs.parameters." + value + "}}"))
                dag_inputs.append(Parameter(name=value))

        clean_name = self.executor.get_clean_name(self.node)
        fan_out_template = self.executor._create_fan_out_template(
            composite_node=self.node, list_of_iter_values=list_of_iter_values
        )
        fan_out_template.arguments = task_template_arguments if task_template_arguments else None

        fan_in_template = self.executor._create_fan_in_template(
            composite_node=self.node, list_of_iter_values=list_of_iter_values
        )
        fan_in_template.arguments = task_template_arguments if task_template_arguments else None

        self.executor._gather_task_templates_of_dag(
            self.node.branch,
            dag_name=f"{clean_name}-branch",
            list_of_iter_values=list_of_iter_values,
        )

        branch_template = TaskTemplate(
            name=f"{clean_name}-branch",
            template=f"{clean_name}-branch",
            arguments=task_template_arguments if task_template_arguments else None,
        )
        branch_template.depends.append(f"{clean_name}-fan-out.Succeeded")
        fan_in_template.depends.append(f"{clean_name}-branch.Succeeded")
        fan_in_template.depends.append(f"{clean_name}-branch.Failed")

        self.executor._dag_templates.append(
            DagTemplate(
                tasks=[fan_out_template, branch_template, fan_in_template],
                name=clean_name,
                inputs=dag_inputs if dag_inputs else None,
            )
        )


class ParallelNodeRender(NodeRenderer):
    allowed_node_types = ["parallel"]

    def render(self, list_of_iter_values: Optional[List] = None):
        self.node = cast(ParallelNode, self.node)
        task_template_arguments = []
        dag_inputs = []
        if list_of_iter_values:
            for value in list_of_iter_values:
                task_template_arguments.append(Argument(name=value, value="{{inputs.parameters." + value + "}}"))
                dag_inputs.append(Parameter(name=value))

        clean_name = self.executor.get_clean_name(self.node)
        fan_out_template = self.executor._create_fan_out_template(
            composite_node=self.node, list_of_iter_values=list_of_iter_values
        )
        fan_out_template.arguments = task_template_arguments if task_template_arguments else None

        fan_in_template = self.executor._create_fan_in_template(
            composite_node=self.node, list_of_iter_values=list_of_iter_values
        )
        fan_in_template.arguments = task_template_arguments if task_template_arguments else None

        branch_templates = []
        for name, branch in self.node.branches.items():
            branch_name = self.executor.sanitize_name(name)
            self.executor._gather_task_templates_of_dag(
                branch,
                dag_name=f"{clean_name}-{branch_name}",
                list_of_iter_values=list_of_iter_values,
            )
            task_template = TaskTemplate(
                name=f"{clean_name}-{branch_name}",
                template=f"{clean_name}-{branch_name}",
                arguments=task_template_arguments if task_template_arguments else None,
            )
            task_template.depends.append(f"{clean_name}-fan-out.Succeeded")
            fan_in_template.depends.append(f"{task_template.name}.Succeeded")
            fan_in_template.depends.append(f"{task_template.name}.Failed")
            branch_templates.append(task_template)

        self.executor._dag_templates.append(
            DagTemplate(
                tasks=[fan_out_template] + branch_templates + [fan_in_template],
                name=clean_name,
                inputs=dag_inputs if dag_inputs else None,
            )
        )


class MapNodeRender(NodeRenderer):
    allowed_node_types = ["map"]

    def render(self, list_of_iter_values: Optional[List] = None):
        self.node = cast(MapNode, self.node)
        task_template_arguments = []
        dag_inputs = []
        if list_of_iter_values:
            for value in list_of_iter_values:
                task_template_arguments.append(Argument(name=value, value="{{inputs.parameters." + value + "}}"))
            dag_inputs.append(Parameter(name=value))

        clean_name = self.executor.get_clean_name(self.node)
        fan_out_template = self.executor._create_fan_out_template(
            composite_node=self.node, list_of_iter_values=list_of_iter_values
        )
        fan_out_template.arguments = task_template_arguments if task_template_arguments else None

        fan_in_template = self.executor._create_fan_in_template(
            composite_node=self.node, list_of_iter_values=list_of_iter_values
        )
        fan_in_template.arguments = task_template_arguments if task_template_arguments else None

        if not list_of_iter_values:
            list_of_iter_values = []

        list_of_iter_values.append(self.node.iterate_as)

        self.executor._gather_task_templates_of_dag(
            self.node.branch,
            dag_name=f"{clean_name}-map",
            list_of_iter_values=list_of_iter_values,
        )

        task_template = TaskTemplate(
            name=f"{clean_name}-map",
            template=f"{clean_name}-map",
            arguments=task_template_arguments if task_template_arguments else None,
        )
        task_template.with_param = "{{tasks." + f"{clean_name}-fan-out" + ".outputs.parameters." + "iterate-on" + "}}"

        argument = Argument(name=self.node.iterate_as, value="{{item}}")
        if task_template.arguments is None:
            task_template.arguments = []
        task_template.arguments.append(argument)

        task_template.depends.append(f"{clean_name}-fan-out.Succeeded")
        fan_in_template.depends.append(f"{clean_name}-map.Succeeded")
        fan_in_template.depends.append(f"{clean_name}-map.Failed")

        executor_config = self.executor._resolve_executor_config(self.node)

        self.executor._dag_templates.append(
            DagTemplate(
                tasks=[fan_out_template, task_template, fan_in_template],
                name=clean_name,
                inputs=dag_inputs if dag_inputs else None,
                parallelism=executor_config.get("parallelism", 0),
                fail_fast=executor_config.get("fail_fast", True),
            )
        )


def get_renderer(node):
    renderers = NodeRenderer.__subclasses__()

    for renderer in renderers:
        if node.node_type in renderer.allowed_node_types:
            return renderer
    raise Exception("This node type is not render-able")


class ArgoExecutor(GenericExecutor, UserControls):
    service_name: str = "argo"

    image: str

    run_id_placeholder: str = "{{workflow.parameters.run_id}}"
    parallelism: Optional[int] = Field(default=None)
    service_account_name: Optional[str] = None
    secrets_from_k8s: List[SecretEnvVar] = Field(default=[])

    persistent_volumes: List[UserVolumeMounts] = []
    fail_fast: bool = True
    expose_parameters_as_inputs: bool = True

    output_file: str = "argo-pipeline.yaml"
    model_config = ConfigDict(extra="forbid")

    _container_templates: List[ContainerTemplate] = []
    _dag_templates: List[DagTemplate] = []
    _clean_names: Dict[str, str] = {}
    _container_volumes: List[ContainerVolume] = []

    def prepare_for_graph_execution(self):
        """
        This method should be called prior to calling execute_graph.
        Perform any steps required before doing the graph execution.

        The most common implementation is to prepare a run log for the run if the run uses local interactive compute.

        But in cases of actual rendering the job specs (eg: AWS step functions, K8's) we check if the services are OK.
        We do not set up a run log as its not relevant.
        """

        integration.validate(self, self._context.run_log_store)
        integration.configure_for_traversal(self, self._context.run_log_store)

        integration.validate(self, self._context.catalog_handler)
        integration.configure_for_traversal(self, self._context.catalog_handler)

        integration.validate(self, self._context.secrets_handler)
        integration.configure_for_traversal(self, self._context.secrets_handler)

        integration.validate(self, self._context.experiment_tracker)
        integration.configure_for_traversal(self, self._context.experiment_tracker)

    def prepare_for_node_execution(self):
        """
        Perform any modifications to the services prior to execution of the node.

        Args:
            node (Node): [description]
            map_variable (dict, optional): [description]. Defaults to None.
        """

        super().prepare_for_node_execution()
        self._set_up_run_log(exists_ok=True)

    def execute_node(self, node: BaseNode, map_variable: Optional[dict] = None, **kwargs):
        step_log = self._context.run_log_store.create_step_log(node.name, node._get_step_log_name(map_variable))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        super()._execute_node(node, map_variable=map_variable, **kwargs)

        # Implicit fail
        if self._context.dag:
            # functions and notebooks do not have dags
            _, current_branch = search_node_by_internal_name(dag=self._context.dag, internal_name=node.internal_name)
            _, next_node_name = self._get_status_and_next_node_name(node, current_branch, map_variable=map_variable)
            if next_node_name:
                # Terminal nodes do not have next node name
                next_node = current_branch.get_node_by_name(next_node_name)

                if next_node.node_type == defaults.FAIL:
                    self.execute_node(next_node, map_variable=map_variable)

        step_log = self._context.run_log_store.get_step_log(node._get_step_log_name(map_variable), self._context.run_id)
        if step_log.status == defaults.FAIL:
            raise Exception(f"Step {node.name} failed")

    def fan_out(self, node: BaseNode, map_variable: Optional[Dict[str, str]] = None):
        super().fan_out(node, map_variable)

        # If its a map node, write the list values to "/tmp/output.txt"
        if node.node_type == "map":
            node = cast(MapNode, node)
            iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[node.iterate_on]

            with open("/tmp/output.txt", mode="w", encoding="utf-8") as myfile:
                json.dump(iterate_on, myfile, indent=4)

    def _get_parameters(self) -> Dict[str, Any]:
        parameters = {}
        if self._context.parameters_file:
            # Parameters from the parameters file if defined
            parameters.update(utils.load_yaml(self._context.parameters_file))
        # parameters from environment variables supersede file based
        parameters.update(utils.get_user_set_parameters())

        return parameters

    def sanitize_name(self, name):
        return name.replace(" ", "-").replace(".", "-").replace("_", "-")

    def get_clean_name(self, node: BaseNode):
        # Cache names for the node
        if node.internal_name not in self._clean_names:
            sanitized = self.sanitize_name(node.name)
            tag = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
            self._clean_names[node.internal_name] = f"{sanitized}-{node.node_type}-{tag}"

        return self._clean_names[node.internal_name]

    def compose_map_variable(self, list_of_iter_values: Optional[List] = None) -> OrderedDict:
        map_variable = OrderedDict()

        # If we are inside a map node, compose a map_variable
        # The values of "iterate_as" are sent over as inputs to the container template
        if list_of_iter_values:
            for var in list_of_iter_values:
                map_variable[var] = "{{inputs.parameters." + str(var) + "}}"

        return map_variable

    def create_container_template(
        self,
        working_on: BaseNode,
        command: str,
        inputs: Optional[List] = None,
        outputs: Optional[List] = None,
        overwrite_name: str = "",
    ):
        effective_node_config = self._resolve_executor_config(working_on)

        default_user_controls = self._get_template_defaults()
        step_controls = UserControls(**effective_node_config)

        container: Union[Container, DefaultContainer]
        if default_user_controls == step_controls:
            container = DefaultContainer(command=command, image=self.image, volume_mounts=self._container_volumes)
        else:
            container = Container(command=command, volume_mounts=self._container_volumes, **effective_node_config)

        if working_on.name == self._context.dag.start_at and self.expose_parameters_as_inputs:
            for key, value in self._get_parameters().items():
                # Get the value from work flow parameters for dynamic behavior
                if isinstance(value, dict) or isinstance(value, list):
                    continue
                env_var = EnvVar(
                    name=defaults.PARAMETER_PREFIX + key,
                    value="{{workflow.parameters." + key + "}}",
                )
                container._env_vars.append(env_var)

        clean_name = self.get_clean_name(working_on)
        if overwrite_name:
            clean_name = overwrite_name

        container_template = ContainerTemplate(name=clean_name, container=container)

        # inputs are the "iterate_as" value map variables in the same order as they are observed
        # We need to expose the map variables in the command of the container
        if inputs:
            if not container_template.inputs:
                container_template.inputs = []
            container_template.inputs.extend(inputs)

        # The map step fan out would create an output that we should propagate via Argo
        if outputs:
            if not container_template.outputs:
                container_template.outputs = []
            container_template.outputs.extend(outputs)

        return container_template

    def _create_fan_out_template(self, composite_node, list_of_iter_values: Optional[List] = None):
        clean_name = self.get_clean_name(composite_node)
        inputs = []
        # If we are fanning out already map state, we need to send the map variable inside
        # The container template also should be accepting an input parameter
        map_variable = None
        if list_of_iter_values:
            map_variable = self.compose_map_variable(list_of_iter_values=list_of_iter_values)

            for val in list_of_iter_values:
                inputs.append(Parameter(name=val))

        command = utils.get_fan_command(
            mode="out",
            node=composite_node,
            run_id=self.run_id_placeholder,
            map_variable=map_variable,
        )

        outputs = []
        # If the node is a map node, we have to set the output parameters
        # Output is always the step's internal name + iterate-on
        if composite_node.node_type == "map":
            output_parameter = OutputParameter(name="iterate-on")
            outputs.append(output_parameter)

        # Create the node now
        step_config = {"command": command, "type": "task", "next": "dummy"}
        node = create_node(name=f"{clean_name}-fan-out", step_config=step_config)

        container_template = self.create_container_template(
            working_on=node,
            command=command,
            outputs=outputs,
            inputs=inputs,
            overwrite_name=f"{clean_name}-fan-out",
        )

        self._container_templates.append(container_template)
        return TaskTemplate(name=f"{clean_name}-fan-out", template=f"{clean_name}-fan-out")

    def _create_fan_in_template(self, composite_node, list_of_iter_values: Optional[List] = None):
        clean_name = self.get_clean_name(composite_node)
        inputs = []
        # If we are fanning in already map state, we need to send the map variable inside
        # The container template also should be accepting an input parameter
        map_variable = None
        if list_of_iter_values:
            map_variable = self.compose_map_variable(list_of_iter_values=list_of_iter_values)

            for val in list_of_iter_values:
                inputs.append(Parameter(name=val))

        command = utils.get_fan_command(
            mode="in",
            node=composite_node,
            run_id=self.run_id_placeholder,
            map_variable=map_variable,
        )

        step_config = {"command": command, "type": "task", "next": "dummy"}
        node = create_node(name=f"{clean_name}-fan-in", step_config=step_config)
        container_template = self.create_container_template(
            working_on=node,
            command=command,
            inputs=inputs,
            overwrite_name=f"{clean_name}-fan-in",
        )
        self._container_templates.append(container_template)
        clean_name = self.get_clean_name(composite_node)
        return TaskTemplate(name=f"{clean_name}-fan-in", template=f"{clean_name}-fan-in")

    def _gather_task_templates_of_dag(
        self, dag: Graph, dag_name="magnus-dag", list_of_iter_values: Optional[List] = None
    ):
        current_node = dag.start_at
        previous_node = None
        previous_node_template_name = None

        templates: Dict[str, TaskTemplate] = {}

        if not list_of_iter_values:
            list_of_iter_values = []

        while True:
            working_on = dag.get_node_by_name(current_node)
            if previous_node == current_node:
                raise Exception("Potentially running in a infinite loop")

            render_obj = get_renderer(working_on)(executor=self, node=working_on)
            render_obj.render(list_of_iter_values=list_of_iter_values.copy())

            clean_name = self.get_clean_name(working_on)

            # If a task template for clean name exists, retrieve it (could have been created by on_failure)
            template = templates.get(clean_name, TaskTemplate(name=clean_name, template=clean_name))

            # Link the current node to previous node, if the previous node was successful.
            if previous_node:
                template.depends.append(f"{previous_node_template_name}.Succeeded")

            templates[clean_name] = template

            # On failure nodes
            if working_on.node_type not in ["success", "fail"] and working_on._get_on_failure_node():
                failure_node = dag.get_node_by_name(working_on._get_on_failure_node())

                failure_template_name = self.get_clean_name(failure_node)
                # If a task template for clean name exists, retrieve it
                failure_template = templates.get(
                    failure_template_name,
                    TaskTemplate(name=failure_template_name, template=failure_template_name),
                )
                failure_template.depends.append(f"{clean_name}.Failed")

                templates[failure_template_name] = failure_template

            # If we are in a map node, we need to add the values as arguments
            template = templates[clean_name]
            if list_of_iter_values:
                if not template.arguments:
                    template.arguments = []
                for value in list_of_iter_values:
                    template.arguments.append(Argument(name=value, value="{{inputs.parameters." + value + "}}"))

            # Move ahead to the next node
            previous_node = current_node
            previous_node_template_name = self.get_clean_name(working_on)

            if working_on.node_type in ["success", "fail"]:
                break

            current_node = working_on._get_next_node()

        # Add the iteration values as input to dag template
        dag_template = DagTemplate(tasks=list(templates.values()), name=dag_name)
        if list_of_iter_values:
            if not dag_template.inputs:
                dag_template.inputs = []
            dag_template.inputs.extend([Parameter(name=val) for val in list_of_iter_values])

        # Add the dag template to the list of templates
        self._dag_templates.append(dag_template)

    def _get_template_defaults(self) -> UserControls:
        template_defaults_keys = list(UserControls.__fields__.keys())  # type: ignore

        user_provided_config = self.model_dump(by_alias=False)
        template_defaults_dict = {key: user_provided_config[key] for key in template_defaults_keys}

        return UserControls(**template_defaults_dict)

    def execute_graph(self, dag: Graph, map_variable: Optional[dict] = None, **kwargs):
        arguments = []
        # Expose "simple" parameters as workflow arguments for dynamic behavior
        if self.expose_parameters_as_inputs:
            for key, value in self._get_parameters().items():
                if isinstance(value, dict) or isinstance(value, list):
                    continue
                env_var = EnvVar(name=key, value=value)
                arguments.append(env_var)

        run_id_var = EnvVar(name="run_id", value="{{workflow.uid}}")
        arguments.append(run_id_var)

        # TODO: Experimental feature
        original_run_id_var = EnvVar(name="original_run_id")
        arguments.append(original_run_id_var)

        template_defaults = self._get_template_defaults()

        specification = Spec(
            service_account_name=self.service_account_name,
            parallelism=self.parallelism,
            arguments=arguments,
            persistent_volumes=self.persistent_volumes,
            template_defaults=template_defaults,
        )

        for volume in specification.volumes:
            self._container_volumes.append(ContainerVolume(name=volume.name, mount_path=volume.mount_path))

        # Container specifications are globally collected and added at the end.
        # Dag specifications are added as part of the dag traversal.
        templates: List[Union[DagTemplate, ContainerTemplate]] = []
        self._gather_task_templates_of_dag(dag=dag, list_of_iter_values=[])
        templates.extend(self._dag_templates)
        templates.extend(self._container_templates)

        specification.templates = templates

        workflow = Workflow(spec=specification)

        yaml = YAML()
        with open(self.output_file, "w") as f:
            yaml.dump(workflow.model_dump(by_alias=True, exclude_none=True), f)

    def execute_job(self, node: BaseNode):
        """
        Use K8's job instead
        """
        raise NotImplementedError("Use K8's job instead")

    def send_return_code(self, stage="traversal"):
        """
        Convenience function used by pipeline to send return code to the caller of the cli

        Raises:
            Exception: If the pipeline execution failed
        """
        if stage != "traversal":  # traversal does no actual execution, so return code is pointless
            run_id = self._context.run_id

            run_log = self._context.run_log_store.get_run_log_by_id(run_id=run_id, full=False)
            if run_log.status == defaults.FAIL:
                raise exceptions.ExecutionFailedError(run_id)
