import logging
import shlex
from enum import Enum
from typing import Annotated, List, Optional

from kubernetes import client
from kubernetes import config as k8s_config
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer
from pydantic.alias_generators import to_camel

from extensions.job_executor import GenericJobExecutor
from runnable import console, defaults, utils
from runnable.datastore import DataCatalog
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.NAME)


class Operator(str, Enum):
    NOT_IN = "NotIn"
    EXISTS = "Exists"
    DOES_NOT_EXIST = "DoesNotExist"


class RestartPolicy(str, Enum):
    ALWAYS = "Always"
    ON_FAILURE = "OnFailure"
    NEVER = "Never"


class ImagePullPolicy(str, Enum):
    ALWAYS = "Always"
    IF_NOT_PRESENT = "IfNotPresent"
    NEVER = "Never"


class LabelSelectorRequirement(BaseModel):
    key: str
    operator: Operator
    values: list[str]


class LabelSelector(BaseModel):
    match_expressions: list[LabelSelectorRequirement]
    match_labels: dict[str, str]


class ObjectMetaData(BaseModel):
    generate_name: Optional[str]
    annotations: Optional[dict[str, str]]
    namespace: Optional[str] = "default"


class EnvVar(BaseModel):
    name: str
    value: str


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
    limits: Limit = Limit()
    requests: Request = Request()


class VolumeMount(BaseModel):
    name: str
    mount_path: str

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class Container(BaseModel):
    image: str
    env: Optional[list[EnvVar]] = None
    image_pull_policy: ImagePullPolicy = ImagePullPolicy.NEVER
    resources: Resources = Resources()
    volume_mounts: Optional[list[VolumeMount]] = Field(default_factory=list)


class HostPath(BaseModel):
    path: str


class Volume(BaseModel):
    name: str
    host_path: HostPath

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class Spec(BaseModel):
    active_deadline_seconds: int = Field(default=60 * 60 * 2)  # 2 hours
    node_selector: Optional[dict[str, str]] = None
    tolerations: Optional[list[dict[str, str]]] = None
    volumes: Optional[list[Volume]] = Field(default_factory=list)
    service_account_name: Optional[str] = "default"
    restart_policy: RestartPolicy = RestartPolicy.NEVER
    container: Container


class Template(BaseModel):
    spec: Spec
    metadata: Optional[ObjectMetaData] = None


class K8sJobExecutor(GenericJobExecutor):
    service_name: str = "k8s-job"
    config_path: Optional[str] = None
    template: Template
    mock: bool = False
    # The location the mount of .run_log_store is mounted to in minikube
    # ensure that minikube mount $HOME/workspace/runnable/.run_log_store:/volume/run_logs is executed first
    # $HOME/workspace/runnable/.catalog:/volume/catalog
    # Ensure that the docker build is done with eval $(minikube docker-env)
    mini_k8s_run_log_location: str = Field(default="/volume/run_logs/")
    mini_k8s_catalog_location: str = Field(default="/volume/catalog/")

    _is_local: bool = False

    _container_log_location = "/tmp/run_logs/"
    _container_catalog_location = "/tmp/catalog/"
    _container_secrets_location = "/tmp/dotenv"

    _volumes: list[Volume] = []
    _volume_mounts: list[VolumeMount] = []

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )

    def submit_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
        """
        This method gets invoked by the CLI.
        """
        self._set_up_run_log()

        # Call the container job
        job_log = self._context.run_log_store.create_job_log()
        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )
        # create volumes and volume mounts for the job
        self._create_volumes()
        self.submit_k8s_job(job)

    def execute_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
        """
        Focusses on execution of the job.
        """
        logger.info("Trying to execute job")
        self._use_volumes()

        job_log = self._context.run_log_store.get_job_log(run_id=self._context.run_id)

        attempt_log = job.execute_command(
            attempt_number=self.step_attempt_number,
            mock=self.mock,
        )

        job_log.status = attempt_log.status
        job_log.attempts.append(attempt_log)

        data_catalogs_put: List[DataCatalog] = self._sync_catalog(
            catalog_settings=catalog_settings
        )
        logger.debug(f"data_catalogs_put: {data_catalogs_put}")

        job_log.add_data_catalogs(data_catalogs_put or [])

        console.print("Summary of job")
        console.print(job_log.get_summary())

        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )

    @property
    def _client(self):
        if self.config_path:
            k8s_config.load_kube_config(config_file=self.config_path)
        else:
            # https://github.com/kubernetes-client/python/blob/master/kubernetes/base/config/__init__.py
            k8s_config.load_config()
        return client

    def submit_k8s_job(self, task: BaseTaskType):
        print(self.model_dump())

        if self.template.spec.container.volume_mounts:
            self._volume_mounts += self.template.spec.container.volume_mounts

        command = utils.get_job_execution_command()

        base_container = self._client.V1Container(
            name="default",
            command=shlex.split(command),
            volume_mounts=[
                vol.model_dump(by_alias=True) for vol in self._volume_mounts
            ],
            **self.template.spec.container.model_dump(
                exclude_none=True, exclude={"volume_mounts", "command"}
            ),
        )

        if self.template.spec.volumes:
            self._volumes += self.template.spec.volumes

        pod_spec = self._client.V1PodSpec(
            containers=[base_container],
            volumes=[vol.model_dump(by_alias=True) for vol in self._volumes],
            **self.template.spec.model_dump(
                exclude_none=True, exclude={"container", "volumes"}
            ),
        )

        pod_template = self._client.V1PodTemplateSpec(
            spec=pod_spec,
            **self.template.model_dump(exclude_none=True, exclude={"spec"}),
        )

        job_spec = client.V1JobSpec(
            template=pod_template,
        )

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=self._context.run_id),
            spec=job_spec,
        )

        print(job.__dict__)

        try:
            k8s_batch = self._client.BatchV1Api()
            response = k8s_batch.create_namespaced_job(
                body=job,
                namespace="default",
                _preload_content=False,
                pretty=True,
            )
            logger.debug(f"Kubernetes job response: {response}")
        except Exception as e:
            logger.exception(e)
            print(e)
            raise

    def _create_volumes(self):
        match self._context.run_log_store.service_name:
            case "file-system":
                self._volumes.append(
                    # When you do: # minikube mount $HOME:/tmp/run_logs
                    # This .run_log_store is mounted to /tmp/run_logs of minikube
                    # You then are creating a volume that is mounted to /tmp/run_logs in the container
                    # You are then referring to it.
                    # https://stackoverflow.com/questions/57411456/minikube-mounted-host-folders-are-not-working
                    Volume(
                        name="run-logs",
                        host_path=HostPath(path=self.mini_k8s_run_log_location),
                    )
                )
                self._volume_mounts.append(
                    VolumeMount(
                        name="run-logs", mount_path=self._container_log_location
                    )
                )
            case "chunked-fs":
                self._volumes.append(
                    Volume(
                        name="run-logs",
                        host_path=HostPath(path=self.mini_k8s_run_log_location),
                    )
                )
                self._volume_mounts.append(
                    VolumeMount(
                        name="run-logs", mount_path=self._container_log_location
                    )
                )

        match self._context.catalog_handler.service_name:
            case "file-system":
                self._volumes.append(
                    Volume(
                        name="catalog",
                        host_path=HostPath(path=self.mini_k8s_catalog_location),
                    )
                )
                self._volume_mounts.append(
                    VolumeMount(
                        name="catalog", mount_path=self._container_catalog_location
                    )
                )

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
