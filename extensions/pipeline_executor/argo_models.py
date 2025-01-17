import random
import shlex
import string
from collections import namedtuple
from enum import Enum
from functools import cached_property
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    computed_field,
    model_validator,
)
from pydantic.alias_generators import to_camel
from ruamel.yaml import YAML

from extensions.nodes.nodes import TaskNode
from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import defaults, utils
from runnable.defaults import TypeMapVariable
from runnable.graph import Graph
from runnable.nodes import BaseNode


class BaseModelWIthConfig(BaseModel, use_enum_values=True):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        validate_default=True,
    )


class BackOff(BaseModelWIthConfig):
    duration: str = Field(default="2m")
    factor: float = Field(default=2)
    max_duration: str = Field(default="1h")


class RetryStrategy(BaseModelWIthConfig):
    back_off: Optional[BackOff] = Field(default=None)
    limit: int = 0
    retry_policy: str = Field(default="Always")


class Parameter(BaseModelWIthConfig):
    name: str
    value: str | int | float | bool


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
    def value_from(self) -> dict[str, dict[str, str]]:
        return {
            "secretKeyRef": {
                "name": self.secret_name,
                "key": self.secret_key,
            }
        }


class EnvVar(BaseModelWIthConfig):
    name: str
    value: str


class Inputs(BaseModelWIthConfig):
    parameters: Optional[list[Parameter]] = Field(default=None)


class Outputs(BaseModelWIthConfig):
    parameters: Optional[list[Parameter]] = Field(default=None)


class Arguments(BaseModelWIthConfig):
    parameters: Optional[list[Parameter]] = Field(default=None)


class TolerationEffect(str, Enum):
    NoSchedule = "NoSchedule"
    PreferNoSchedule = "PreferNoSchedule"
    NoExecute = "NoExecute"


class TolerationOperator(str, Enum):
    Exists = "Exists"
    Equal = "Equal"


class PodMetaData(BaseModelWIthConfig):
    annotations: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)


class Toleration(BaseModelWIthConfig):
    effect: Optional[TolerationEffect] = Field(default=None)
    key: Optional[str] = Field(default=None)
    operator: TolerationOperator = Field(default=TolerationOperator.Equal)
    tolerationSeconds: Optional[int] = Field(default=None)
    value: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_tolerations(self) -> "Toleration":
        if not self.key:
            if self.operator != TolerationOperator.Exists:
                raise ValueError("Toleration key is required when operator is Equal")

        if self.operator == TolerationOperator.Exists:
            if self.value:
                raise ValueError(
                    "Toleration value is not allowed when operator is Exists"
                )
        return self


class ImagePullPolicy(str, Enum):
    Always = "Always"
    IfNotPresent = "IfNotPresent"
    Never = "Never"


class PersistentVolumeClaimSource(BaseModelWIthConfig):
    claim_name: str
    read_only: bool = Field(default=False)


class Volume(BaseModelWIthConfig):
    name: str
    persistent_volume_claim: PersistentVolumeClaimSource

    def __hash__(self):
        return hash(self.name)


class VolumeMount(BaseModelWIthConfig):
    mount_path: str
    name: str
    read_only: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_volume_mount(self) -> "VolumeMount":
        if "." in self.mount_path:
            raise ValueError("mount_path cannot contain '.'")

        return self


VolumePair = namedtuple("VolumePair", ["volume", "volume_mount"])


class LabelSelectorRequirement(BaseModelWIthConfig):
    key: str
    operator: str
    values: list[str]


class PodGCStrategy(str, Enum):
    OnPodCompletion = "OnPodCompletion"
    OnPodSuccess = "OnPodSuccess"
    OnWorkflowCompletion = "OnWorkflowCompletion"
    OnWorkflowSuccess = "OnWorkflowSuccess"


class LabelSelector(BaseModelWIthConfig):
    matchExpressions: list[LabelSelectorRequirement] = Field(default_factory=list)
    matchLabels: dict[str, str] = Field(default_factory=dict)


class PodGC(BaseModelWIthConfig):
    delete_delay_duration: str = Field(default="1h")  # 1 hour
    label_selector: Optional[LabelSelector] = Field(default=None)
    strategy: Optional[PodGCStrategy] = Field(default=None)


class Request(BaseModel):
    """
    The default requests
    """

    memory: str = "1Gi"
    cpu: str = "250m"


VendorGPU = Annotated[
    Optional[int],
    PlainSerializer(lambda x: str(x), return_type=str, when_used="unless-none"),
]


class Limit(Request):
    """
    The default limits
    """

    gpu: VendorGPU = Field(default=None, serialization_alias="nvidia.com/gpu")


class Resources(BaseModel):
    limits: Limit = Field(default=Limit(), serialization_alias="limits")
    requests: Request = Field(default=Request(), serialization_alias="requests")


# This is what the user can override per template
# Some are specific to container and some are specific to dag
class TemplateDefaults(BaseModelWIthConfig):
    active_deadline_seconds: Optional[int] = Field(default=86400)  # 1 day
    fail_fast: bool = Field(default=True)
    node_selector: dict[str, str] = Field(default_factory=dict)
    parallelism: Optional[int] = Field(default=None)
    retry_strategy: Optional[RetryStrategy] = Field(default=None)
    timeout: Optional[str] = Field(default=None)
    tolerations: Optional[list[Toleration]] = Field(default=None)

    # These are in addition to what argo spec provides
    image: str
    image_pull_policy: Optional[ImagePullPolicy] = Field(default=ImagePullPolicy.Always)
    resources: Resources = Field(default_factory=Resources)


# TODO: Can user provide env vars for the container?


# User provides this as part of the argoSpec
# some an be provided here or as a template default or node override
class ArgoWorkflowSpec(BaseModelWIthConfig):
    active_deadline_seconds: int = Field(default=86400)  # 1 day for the whole workflow
    arguments: Optional[Arguments] = Field(default=None)
    entrypoint: Literal["runnable-dag"] = Field(default="runnable-dag", frozen=True)
    node_selector: dict[str, str] = Field(default_factory=dict)
    parallelism: Optional[int] = Field(default=None)  # GLobal parallelism
    pod_gc: Optional[PodGC] = Field(default=None, serialization_alias="podGC")
    retry_strategy: Optional[RetryStrategy] = Field(default=None)
    service_account_name: Optional[str] = Field(default=None)
    template_defaults: TemplateDefaults = Field(default_factory=TemplateDefaults)
    tolerations: Optional[list[Toleration]] = Field(default=None)


class ArgoMetadata(BaseModelWIthConfig):
    annotations: Optional[dict[str, str]] = Field(default=None)
    generate_name: str  # User can mention this to uniquely identify the run
    labels: dict[str, str] = Field(default_factory=dict)
    namespace: Optional[str] = Field(default="default")


class ArgoWorkflow(BaseModelWIthConfig):
    apiVersion: Literal["argoproj.io/v1alpha1"] = Field(
        default="argoproj.io/v1alpha1", frozen=True
    )
    kind: Literal["Workflow"] = Field(default="Workflow", frozen=True)
    metadata: ArgoMetadata
    spec: ArgoWorkflowSpec = Field(default_factory=ArgoWorkflowSpec)


# The below are not visible to the user
class DagTask(BaseModelWIthConfig):
    arguments: Optional[Arguments] = Field(default=None)
    depends: Optional[str] = Field(default=None)
    name: str
    template: str  # Should be name of a container template or dag template
    with_params: Optional[str] = Field(default=None)


class CoreDagTemplate(BaseModelWIthConfig):
    tasks: list[DagTask] = Field(default_factory=list)


class CoreContainerTemplate(BaseModelWIthConfig):
    image: str
    command: list[str]
    image_pull_policy: ImagePullPolicy = Field(default=ImagePullPolicy.IfNotPresent)
    env: list[EnvVar | SecretEnvVar] = Field(default_factory=list)
    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    resources: Resources = Field(default_factory=Resources)


class DagTemplate(BaseModelWIthConfig):
    name: str
    dag: Optional[CoreDagTemplate] = Field(default=None)  # Should be filled in
    inputs: Optional[Inputs] = Field(default=None)
    outputs: Optional[Outputs] = Field(default=None)
    parallelism: Optional[int] = Field(default=None)  # Not sure if this is needed
    fail_fast: bool = Field(default=True)

    model_config = ConfigDict(
        extra="ignore",
    )


class ContainerTemplate((BaseModelWIthConfig)):
    name: str
    container: CoreContainerTemplate
    inputs: Optional[Inputs] = Field(default=None)
    outputs: Optional[Outputs] = Field(default=None)

    # The remaining can be from template defaults or node overrides
    active_deadline_seconds: Optional[int] = Field(default=86400)  # 1 day
    metadata: Optional[PodMetaData] = Field(default=None)
    node_selector: dict[str, str] = Field(default_factory=dict)
    parallelism: Optional[int] = Field(default=None)  # Not sure if this is needed
    retry_strategy: Optional[RetryStrategy] = Field(default=None)
    timeout: Optional[str] = Field(default=None)
    tolerations: Optional[list[Toleration]] = Field(default=None)
    volumes: Optional[list[Volume]] = Field(default=None)

    model_config = ConfigDict(
        extra="ignore",
    )


class CustomVolume(BaseModelWIthConfig):
    mount_path: str
    persistent_volume_claim: PersistentVolumeClaimSource


class ArgoExecutor(GenericPipelineExecutor):
    service_name: str = "argo"
    _is_local: bool = False
    mock: bool = False

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        use_enum_values=True,
    )

    argo_workflow: ArgoWorkflow

    # Lets use a generic one
    pvc_for_runnable: Optional[str] = Field(default=None)
    # pvc_for_catalog: Optional[str] = Field(default=None)
    # pvc_for_run_log: Optional[str] = Field(default=None)
    custom_volumes: Optional[list[CustomVolume]] = Field(default_factory=list)

    expose_parameters_as_inputs: bool = True
    secret_from_k8s: Optional[str] = Field(default=None)
    output_file: str = Field(default="argo-pipeline.yaml")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # This should be used when we refer to run_id or log_level in the containers
    _run_id_as_parameter: str = PrivateAttr(default="{{workflow.parameters.run_id}}")
    _log_level_as_parameter: str = PrivateAttr(
        default="{{workflow.parameters.log_level}}"
    )

    _templates: list[ContainerTemplate | DagTemplate] = PrivateAttr(
        default_factory=list
    )
    _container_log_location: str = PrivateAttr(default="/tmp/run_logs/")
    _container_catalog_location: str = PrivateAttr(default="/tmp/catalog/")

    def sanitize_name(self, name: str, node_type: str) -> str:
        formatted_name = name.replace(" ", "-").replace(".", "-").replace("_", "-")
        tag = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        unique_name = f"{formatted_name}-{node_type}-{tag}"
        return unique_name

    def _create_container_template(
        self,
        node: BaseNode,
        task_name: str,
        map_variable: Optional[TypeMapVariable],
    ) -> ContainerTemplate:
        template_defaults = self.argo_workflow.spec.template_defaults.model_dump()

        node_overide = {}
        if hasattr(node, "overides"):
            node_overide = node.overides  # type: ignore

        # update template defaults with node overrides
        template_defaults.update(node_overide)

        # command = "runnable execute-single-node"
        command = utils.get_node_execution_command(
            node=node,
            map_variable=map_variable,
            over_write_run_id=self._run_id_as_parameter,
            log_level=self._log_level_as_parameter,
        )

        core_container_template = CoreContainerTemplate(
            command=shlex.split(command),
            image=template_defaults["image"],
            image_pull_policy=template_defaults["image_pull_policy"],
            volume_mounts=[
                volume_pair.volume_mount for volume_pair in self.volume_pairs
            ],
        )

        container_template = ContainerTemplate(
            container=core_container_template,
            name=task_name,
            volumes=[volume_pair.volume for volume_pair in self.volume_pairs],
            **template_defaults,
        )

        return container_template

    def _add_env_vars_to_container_template(
        self,
        working_on: BaseNode,
        container_template: CoreContainerTemplate,
        add_override_parameters: bool = False,
    ):
        # TODO:  Add map variables too if agreed
        # TODO: Instead of add_overrides take in a initial_container flag
        assert isinstance(working_on, TaskNode)
        parameters = {}
        if not self.argo_workflow.spec.arguments:
            parameters = self.argo_workflow.spec.arguments.parameters

        if add_override_parameters:
            for parameter in parameters or []:
                key, _ = parameter.name, parameter.value
                env_var = EnvVar(
                    name=defaults.PARAMETER_PREFIX + key,
                    value="{{workflow.parameters." + key + "}}",
                )
                container_template.env.append(env_var)

        secrets = working_on.executable.secrets
        for secret in secrets:
            assert self.secret_from_k8s is not None
            secret_env_var = SecretEnvVar(
                environment_variable=secret,
                secret_name=self.secret_from_k8s,  # This has to be exposed from config
                secret_key=secret,
            )
            container_template.env.append(secret_env_var)

    def _gather_tasks_for_dag_template(
        self,
        dag_template: DagTemplate,
        dag: Graph,
        map_variable: Optional[TypeMapVariable] = None,
        override_parameters: bool = False,
    ):
        # TODO: Handle map variable complications
        """
        Iterate through the graph
        - If the node is not composite
            - Create a container template
            - Create a task
        """
        current_node: str = dag.start_at
        depends: str = ""

        dag_template.dag = CoreDagTemplate()

        while True:
            working_on: BaseNode = dag.get_node_by_name(current_node)
            match working_on.node_type:
                case "task" | "success" | "stub":
                    task_name = self.sanitize_name(working_on.internal_name, "task")
                    template_of_container = self._create_container_template(
                        working_on,
                        task_name=task_name,
                        map_variable=map_variable,
                    )
                    assert template_of_container.container is not None

                    if working_on.node_type == "task":
                        self._add_env_vars_to_container_template(
                            working_on,
                            container_template=template_of_container.container,
                            add_override_parameters=override_parameters,
                        )

                    current_task = DagTask(
                        name=task_name,
                        template=task_name,
                        depends=depends if not depends else depends + ".Succeeded",
                    )

                    dag_template.dag.tasks.append(current_task)
                    depends = task_name
                    self._templates.append(template_of_container)
                case "parallel":
                    # A fan out execution
                    # Create a new dag template
                    # A fan in execution
                    ...
                case "map":
                    # A fan out execution
                    # Create a new dag template
                    # A fan in execution
                    ...

            if working_on.node_type == "success" or working_on.node_type == "fail":
                break
            override_parameters: bool = False
            current_node = working_on._get_next_node()

        self._templates.append(dag_template)

    def execute_graph(
        self,
        dag: Graph,
        map_variable: dict[str, str | int | float] | None = None,
        **kwargs,
    ):
        # All the arguments set at the spec level can be referred as "{{workflow.parameters.*}}"
        # We want to use that functionality to override the parameters at the task level
        # We should be careful to override them only at the first task.
        arguments = []  # Can be updated in the UI
        if self.expose_parameters_as_inputs:
            for key, value in self._get_parameters().items():
                value = value.get_value()  # type: ignore
                if isinstance(value, dict) or isinstance(value, list):
                    continue

                parameter = Parameter(name=key, value=value)  # type: ignore
                arguments.append(parameter)

        run_id_var = Parameter(name="run_id", value="{{workflow.uid}}")
        log_level_var = Parameter(name="log_level", value=self.log_level)
        arguments.append(run_id_var)
        arguments.append(log_level_var)
        self.argo_workflow.spec.arguments = Arguments(parameters=arguments)

        # This is the entry point of the argo execution
        runnable_dag: DagTemplate = DagTemplate(name="runnable-dag")
        self._gather_tasks_for_dag_template(
            runnable_dag, dag, override_parameters=self.expose_parameters_as_inputs
        )

        argo_workflow_dump = self.argo_workflow.model_dump(
            by_alias=True,
            exclude={
                "spec": {
                    "template_defaults": {"image_pull_policy", "image", "resources"}
                }
            },
            exclude_none=True,
            round_trip=False,
        )
        argo_workflow_dump["spec"]["templates"] = [
            template.model_dump(
                by_alias=True,
                exclude_none=True,
            )
            for template in self._templates
        ]

        argo_workflow_dump["spec"]["volumes"] = [
            volume_pair.volume.model_dump(by_alias=True)
            for volume_pair in self.volume_pairs
        ]

        yaml = YAML()
        with open(self.output_file, "w") as f:
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.dump(
                argo_workflow_dump,
                f,
            )

    def execute_node(
        self,
        node: BaseNode,
        map_variable: dict[str, str | int | float] | None = None,
        **kwargs,
    ):
        # TODO: Having an optional flag to set up run log might be a good idea
        # This should only be for the first step of the graph
        self._use_volumes()
        self._set_up_run_log(exists_ok=True)

        step_log = self._context.run_log_store.create_step_log(
            node.name, node._get_step_log_name(map_variable)
        )

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        self._execute_node(node=node, map_variable=map_variable, **kwargs)

        # TODO: Implicit failure handling
        # self.send_return_code()

    def _use_volumes(self):
        match self._context.run_log_store.service_name:
            case "file-system":
                self._context.run_log_store.log_folder = self._container_log_location
            case "chunked-fs":
                self._context.run_log_store.log_folder = self._container_log_location

        match self._context.catalog_handler.service_name:
            case "file-system":
                self._context.catalog_handler.catalog_location = (
                    self._container_catalog_location
                )

    @cached_property
    def volume_pairs(self) -> list[VolumePair]:
        volume_pairs: list[VolumePair] = []

        if self.pvc_for_runnable:
            common_volume = Volume(
                name="runnable",
                persistent_volume_claim=PersistentVolumeClaimSource(
                    claim_name=self.pvc_for_runnable
                ),
            )
            common_volume_mount = VolumeMount(
                name="runnable",
                mount_path="/tmp",
            )
            volume_pairs.append(
                VolumePair(volume=common_volume, volume_mount=common_volume_mount)
            )
        counter = 0
        for custom_volume in self.custom_volumes or []:
            name = f"custom-volume-{counter}"
            volume_pairs.append(
                VolumePair(
                    volume=Volume(
                        name=name,
                        persistent_volume_claim=custom_volume.persistent_volume_claim,
                    ),
                    volume_mount=VolumeMount(
                        name=name,
                        mount_path=custom_volume.mount_path,
                    ),
                )
            )
            counter += 1
        return volume_pairs
