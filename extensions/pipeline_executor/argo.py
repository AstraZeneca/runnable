import json
import os
import random
import shlex
import string
from collections import namedtuple
from enum import Enum
from functools import cached_property
from typing import Annotated, Any, Literal, Optional, cast

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

from extensions.nodes.nodes import (
    MapNode,
    ParallelNode,
    StubNode,
    SuccessNode,
    TaskNode,
)
from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import defaults, utils
from runnable.defaults import TypeMapVariable
from runnable.graph import Graph, search_node_by_internal_name
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


class OutputParameter(BaseModelWIthConfig):
    name: str
    value_from: dict[str, str] = {
        "path": "/tmp/output.txt",
    }


class Parameter(BaseModelWIthConfig):
    name: str
    value: Optional[str | int | float | bool] = Field(default=None)


class Inputs(BaseModelWIthConfig):
    parameters: Optional[list[Parameter]] = Field(default=None)


class Outputs(BaseModelWIthConfig):
    parameters: Optional[list[OutputParameter]] = Field(default=None)


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


# Lets construct this from UserDefaults
class ArgoTemplateDefaults(BaseModelWIthConfig):
    active_deadline_seconds: Optional[int] = Field(default=86400)  # 1 day
    fail_fast: bool = Field(default=True)
    node_selector: dict[str, str] = Field(default_factory=dict)
    parallelism: Optional[int] = Field(default=None)
    retry_strategy: Optional[RetryStrategy] = Field(default=None)
    timeout: Optional[str] = Field(default=None)
    tolerations: Optional[list[Toleration]] = Field(default=None)

    model_config = ConfigDict(
        extra="ignore",
    )


class CommonDefaults(BaseModelWIthConfig):
    active_deadline_seconds: Optional[int] = Field(default=86400)  # 1 day
    fail_fast: bool = Field(default=True)
    node_selector: dict[str, str] = Field(default_factory=dict)
    parallelism: Optional[int] = Field(default=None)
    retry_strategy: Optional[RetryStrategy] = Field(default=None)
    timeout: Optional[str] = Field(default=None)
    tolerations: Optional[list[Toleration]] = Field(default=None)
    image_pull_policy: ImagePullPolicy = Field(default=ImagePullPolicy.Always)
    resources: Resources = Field(default_factory=Resources)
    env: list[EnvVar | SecretEnvVar] = Field(default_factory=list, exclude=True)


# The user provided defaults at the top level
class UserDefaults(CommonDefaults):
    image: str


# Overrides need not have image
class Overrides(CommonDefaults):
    image: Optional[str] = Field(default=None)


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
    tolerations: Optional[list[Toleration]] = Field(default=None)
    template_defaults: Optional[ArgoTemplateDefaults] = Field(default=None)


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
    spec: ArgoWorkflowSpec


# The below are not visible to the user
class DagTask(BaseModelWIthConfig):
    name: str
    template: str  # Should be name of a container template or dag template
    arguments: Optional[Arguments] = Field(default=None)
    with_param: Optional[str] = Field(default=None)
    depends: Optional[str] = Field(default=None)


class CoreDagTemplate(BaseModelWIthConfig):
    tasks: list[DagTask] = Field(default_factory=list[DagTask])


class CoreContainerTemplate(BaseModelWIthConfig):
    image: str
    command: list[str]
    image_pull_policy: ImagePullPolicy = Field(default=ImagePullPolicy.IfNotPresent)
    env: list[EnvVar | SecretEnvVar] = Field(default_factory=list)
    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    resources: Resources = Field(default_factory=Resources)


class DagTemplate(BaseModelWIthConfig):
    name: str
    dag: CoreDagTemplate = Field(default_factory=CoreDagTemplate)
    inputs: Optional[Inputs] = Field(default=None)
    parallelism: Optional[int] = Field(default=None)  # Not sure if this is needed
    fail_fast: bool = Field(default=True)

    model_config = ConfigDict(
        extra="ignore",
    )

    def __hash__(self):
        return hash(self.name)


class ContainerTemplate((BaseModelWIthConfig)):
    name: str
    container: CoreContainerTemplate
    inputs: Optional[Inputs] = Field(default=None)
    outputs: Optional[Outputs] = Field(default=None)

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

    def __hash__(self):
        return hash(self.name)


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
    pvc_for_runnable: Optional[str] = Field(default=None)
    custom_volumes: Optional[list[CustomVolume]] = Field(
        default_factory=list[CustomVolume]
    )

    expose_parameters_as_inputs: bool = True
    secret_from_k8s: Optional[str] = Field(default=None)
    output_file: str = Field(default="argo-pipeline.yaml")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    defaults: UserDefaults
    argo_workflow: ArgoWorkflow

    overrides: dict[str, Overrides] = Field(default_factory=dict)

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
    _added_initial_container: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: Any) -> None:
        self.argo_workflow.spec.template_defaults = ArgoTemplateDefaults(
            **self.defaults.model_dump()
        )

    def sanitize_name(self, name: str) -> str:
        formatted_name = name.replace(" ", "-").replace(".", "-").replace("_", "-")
        tag = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        unique_name = f"{formatted_name}-{tag}"
        unique_name = unique_name.replace("map-variable-placeholder-", "")
        return unique_name

    def _set_up_initial_container(self, container_template: CoreContainerTemplate):
        if self._added_initial_container:
            return

        parameters: list[Parameter] = []

        if self.argo_workflow.spec.arguments:
            parameters = self.argo_workflow.spec.arguments.parameters or []

        for parameter in parameters or []:
            key, _ = parameter.name, parameter.value
            env_var = EnvVar(
                name=defaults.PARAMETER_PREFIX + key,
                value="{{workflow.parameters." + key + "}}",
            )
            container_template.env.append(env_var)

        env_var = EnvVar(name="error_on_existing_run_id", value="true")
        container_template.env.append(env_var)

        # After the first container is added, set the added_initial_container to True
        self._added_initial_container = True

    def _create_fan_templates(
        self,
        node: BaseNode,
        mode: str,
        parameters: Optional[list[Parameter]],
        task_name: str,
    ):
        map_variable: TypeMapVariable = {}
        for parameter in parameters or []:
            map_variable[parameter.name] = (  # type: ignore
                "{{inputs.parameters." + str(parameter.name) + "}}"
            )

        fan_command = utils.get_fan_command(
            mode=mode,
            node=node,
            run_id=self._run_id_as_parameter,
            map_variable=map_variable,
        )

        core_container_template = CoreContainerTemplate(
            command=shlex.split(fan_command),
            image=self.defaults.image,
            image_pull_policy=self.defaults.image_pull_policy,
            volume_mounts=[
                volume_pair.volume_mount for volume_pair in self.volume_pairs
            ],
        )

        # Either a task or a fan-out can the first container
        self._set_up_initial_container(container_template=core_container_template)

        task_name += f"-fan-{mode}"

        outputs: Optional[Outputs] = None
        if mode == "out" and node.node_type == "map":
            outputs = Outputs(parameters=[OutputParameter(name="iterate-on")])

        container_template = ContainerTemplate(
            name=task_name,
            container=core_container_template,
            inputs=Inputs(parameters=parameters),
            outputs=outputs,
            active_deadline_seconds=self.defaults.active_deadline_seconds,
            node_selector=self.defaults.node_selector,
            parallelism=self.defaults.parallelism,
            retry_strategy=self.defaults.retry_strategy,
            timeout=self.defaults.timeout,
            tolerations=self.defaults.tolerations,
            volumes=[volume_pair.volume for volume_pair in self.volume_pairs],
        )

        self._templates.append(container_template)

    def _create_container_template(
        self,
        node: BaseNode,
        task_name: str,
        inputs: Optional[Inputs] = None,
    ) -> ContainerTemplate:
        assert (
            isinstance(node, TaskNode)
            or isinstance(node, StubNode)
            or isinstance(node, SuccessNode)
        )

        node_override = None
        if hasattr(node, "overrides"):
            override_key = node.overrides.get(self.service_name, "")
            try:
                node_override = self.overrides.get(override_key)
            except:  # noqa
                raise Exception("Override not found for: ", override_key)

        effective_settings = self.defaults.model_dump()
        if node_override:
            effective_settings.update(node_override.model_dump())

        inputs = inputs or Inputs(parameters=[])

        map_variable: TypeMapVariable = {}
        for parameter in inputs.parameters or []:
            map_variable[parameter.name] = (  # type: ignore
                "{{inputs.parameters." + str(parameter.name) + "}}"
            )

        # command = "runnable execute-single-node"
        command = utils.get_node_execution_command(
            node=node,
            over_write_run_id=self._run_id_as_parameter,
            map_variable=map_variable,
            log_level=self._log_level_as_parameter,
        )

        core_container_template = CoreContainerTemplate(
            command=shlex.split(command),
            image=effective_settings["image"],
            image_pull_policy=effective_settings["image_pull_policy"],
            resources=effective_settings["resources"],
            volume_mounts=[
                volume_pair.volume_mount for volume_pair in self.volume_pairs
            ],
        )

        self._set_up_initial_container(container_template=core_container_template)
        self._expose_secrets_to_task(
            working_on=node, container_template=core_container_template
        )
        self._set_env_vars_to_task(node, core_container_template)

        container_template = ContainerTemplate(
            name=task_name,
            container=core_container_template,
            inputs=Inputs(
                parameters=[
                    Parameter(name=param.name) for param in inputs.parameters or []
                ]
            ),
            volumes=[volume_pair.volume for volume_pair in self.volume_pairs],
            **node_override.model_dump() if node_override else {},
        )

        return container_template

    def _set_env_vars_to_task(
        self, working_on: BaseNode, container_template: CoreContainerTemplate
    ):
        if not isinstance(working_on, TaskNode):
            return

        global_envs: dict[str, str] = {}

        for env_var in self.defaults.env:
            env_var = cast(EnvVar, env_var)
            global_envs[env_var.name] = env_var.value

        override_key = working_on.overrides.get(self.service_name, "")
        node_override = self.overrides.get(override_key, None)

        # Update the global envs with the node overrides
        if node_override:
            for env_var in node_override.env:
                env_var = cast(EnvVar, env_var)
                global_envs[env_var.name] = env_var.value

        for key, value in global_envs.items():
            env_var_to_add = EnvVar(name=key, value=value)
            container_template.env.append(env_var_to_add)

    def _expose_secrets_to_task(
        self,
        working_on: BaseNode,
        container_template: CoreContainerTemplate,
    ):
        if not isinstance(working_on, TaskNode):
            return
        secrets = working_on.executable.secrets
        for secret in secrets:
            assert self.secret_from_k8s is not None
            secret_env_var = SecretEnvVar(
                environment_variable=secret,
                secret_name=self.secret_from_k8s,  # This has to be exposed from config
                secret_key=secret,
            )
            container_template.env.append(secret_env_var)

    def _handle_failures(
        self,
        working_on: BaseNode,
        dag: Graph,
        task_name: str,
        parent_dag_template: DagTemplate,
    ):
        if working_on._get_on_failure_node():
            # Create a new dag template
            on_failure_dag: DagTemplate = DagTemplate(name=f"on-failure-{task_name}")
            # Add on failure of the current task to be the failure dag template
            on_failure_task = DagTask(
                name=f"on-failure-{task_name}",
                template=f"on-failure-{task_name}",
                depends=task_name + ".Failed",
            )
            # Set failfast of the dag template to be false
            # If not, this branch will never be invoked
            parent_dag_template.fail_fast = False

            assert parent_dag_template.dag

            parent_dag_template.dag.tasks.append(on_failure_task)
            self._gather_tasks_for_dag_template(
                on_failure_dag,
                dag=dag,
                start_at=working_on._get_on_failure_node(),
            )

    # For the future me:
    # - A task can output a array: in this case, its the fan out.
    # - We are using withParam and arguments of the map template to send that value in
    # - The map template should receive that value as a parameter into the template.
    # - The task then start to use it as inputs.parameters.iterate-on

    def _gather_tasks_for_dag_template(
        self,
        dag_template: DagTemplate,
        dag: Graph,
        start_at: str,
        parameters: Optional[list[Parameter]] = None,
    ):
        current_node: str = start_at
        depends: str = ""

        dag_template.dag = CoreDagTemplate()

        while True:
            # Create the dag task with for the parent dag
            working_on: BaseNode = dag.get_node_by_name(current_node)
            task_name = self.sanitize_name(working_on.internal_name)
            current_task = DagTask(
                name=task_name,
                template=task_name,
                depends=depends if not depends else depends + ".Succeeded",
                arguments=Arguments(
                    parameters=[
                        Parameter(
                            name=param.name,
                            value=f"{{{{inputs.parameters.{param.name}}}}}",
                        )
                        for param in parameters or []
                    ]
                ),
            )
            dag_template.dag.tasks.append(current_task)
            depends = task_name

            match working_on.node_type:
                case "task" | "success" | "stub":
                    template_of_container = self._create_container_template(
                        working_on,
                        task_name=task_name,
                        inputs=Inputs(parameters=parameters),
                    )
                    assert template_of_container.container is not None

                    if working_on.node_type == "task":
                        self._expose_secrets_to_task(
                            working_on,
                            container_template=template_of_container.container,
                        )

                    self._templates.append(template_of_container)

                case "map" | "parallel":
                    assert isinstance(working_on, MapNode) or isinstance(
                        working_on, ParallelNode
                    )
                    node_type = working_on.node_type

                    composite_template: DagTemplate = DagTemplate(
                        name=task_name, fail_fast=False
                    )

                    # Add the fan out task
                    fan_out_task = DagTask(
                        name=f"{task_name}-fan-out",
                        template=f"{task_name}-fan-out",
                        arguments=Arguments(parameters=parameters),
                    )
                    composite_template.dag.tasks.append(fan_out_task)
                    self._create_fan_templates(
                        node=working_on,
                        mode="out",
                        parameters=parameters,
                        task_name=task_name,
                    )

                    # Add the composite task
                    with_param = None
                    added_parameters = parameters or []
                    branches = {}
                    if node_type == "map":
                        # If the node is map, we need to handle the iterate as and on
                        assert isinstance(working_on, MapNode)
                        added_parameters = added_parameters + [
                            Parameter(name=working_on.iterate_as, value="{{item}}")
                        ]
                        with_param = f"{{{{tasks.{task_name}-fan-out.outputs.parameters.iterate-on}}}}"

                        branches["branch"] = working_on.branch
                    elif node_type == "parallel":
                        assert isinstance(working_on, ParallelNode)
                        branches = working_on.branches
                    else:
                        raise ValueError("Invalid node type")

                    fan_in_depends = ""

                    for name, branch in branches.items():
                        name = (
                            name.replace(" ", "-").replace(".", "-").replace("_", "-")
                        )

                        branch_task = DagTask(
                            name=f"{task_name}-{name}",
                            template=f"{task_name}-{name}",
                            depends=f"{task_name}-fan-out.Succeeded",
                            arguments=Arguments(parameters=added_parameters),
                            with_param=with_param,
                        )
                        composite_template.dag.tasks.append(branch_task)

                        branch_template = DagTemplate(
                            name=branch_task.name,
                            inputs=Inputs(
                                parameters=[
                                    Parameter(name=param.name, value=None)
                                    for param in added_parameters
                                ]
                            ),
                        )

                        self._gather_tasks_for_dag_template(
                            dag_template=branch_template,
                            dag=branch,
                            start_at=branch.start_at,
                            parameters=added_parameters,
                        )

                        fan_in_depends += f"{branch_task.name}.Succeeded || {branch_task.name}.Failed || "

                    fan_in_task = DagTask(
                        name=f"{task_name}-fan-in",
                        template=f"{task_name}-fan-in",
                        depends=fan_in_depends.strip(" || "),
                        arguments=Arguments(parameters=parameters),
                    )

                    composite_template.dag.tasks.append(fan_in_task)
                    self._create_fan_templates(
                        node=working_on,
                        mode="in",
                        parameters=parameters,
                        task_name=task_name,
                    )

                    self._templates.append(composite_template)

            self._handle_failures(
                working_on,
                dag,
                task_name,
                parent_dag_template=dag_template,
            )

            if working_on.node_type == "success" or working_on.node_type == "fail":
                break

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
            runnable_dag,
            dag,
            start_at=dag.start_at,
            parameters=[],
        )

        argo_workflow_dump = self.argo_workflow.model_dump(
            by_alias=True,
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

    def _implicitly_fail(self, node: BaseNode, map_variable: TypeMapVariable):
        assert self._context.dag
        _, current_branch = search_node_by_internal_name(
            dag=self._context.dag, internal_name=node.internal_name
        )
        _, next_node_name = self._get_status_and_next_node_name(
            node, current_branch, map_variable=map_variable
        )
        if next_node_name:
            # Terminal nodes do not have next node name
            next_node = current_branch.get_node_by_name(next_node_name)

            if next_node.node_type == defaults.FAIL:
                self.execute_node(next_node, map_variable=map_variable)

    def execute_node(
        self,
        node: BaseNode,
        map_variable: dict[str, str | int | float] | None = None,
        **kwargs,
    ):
        error_on_existing_run_id = os.environ.get("error_on_existing_run_id", "false")
        exists_ok = error_on_existing_run_id == "false"

        self._use_volumes()
        self._set_up_run_log(exists_ok=exists_ok)

        step_log = self._context.run_log_store.create_step_log(
            node.name, node._get_step_log_name(map_variable)
        )

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        self._execute_node(node=node, map_variable=map_variable, **kwargs)

        # Raise exception if the step failed
        step_log = self._context.run_log_store.get_step_log(
            node._get_step_log_name(map_variable), self._context.run_id
        )
        if step_log.status == defaults.FAIL:
            raise Exception(f"Step {node.name} failed")

        self._implicitly_fail(node, map_variable)

    def fan_out(self, node: BaseNode, map_variable: TypeMapVariable = None):
        # This could be the first step of the graph
        self._use_volumes()

        error_on_existing_run_id = os.environ.get("error_on_existing_run_id", "false")
        exists_ok = error_on_existing_run_id == "false"
        self._set_up_run_log(exists_ok=exists_ok)

        super().fan_out(node, map_variable)

        # If its a map node, write the list values to "/tmp/output.txt"
        if node.node_type == "map":
            assert isinstance(node, MapNode)
            iterate_on = self._context.run_log_store.get_parameters(
                self._context.run_id
            )[node.iterate_on]

            with open("/tmp/output.txt", mode="w", encoding="utf-8") as myfile:
                json.dump(iterate_on.get_value(), myfile, indent=4)

    def fan_in(self, node: BaseNode, map_variable: TypeMapVariable = None):
        self._use_volumes()
        super().fan_in(node, map_variable)

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
