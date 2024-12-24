# import logging
# import shlex
# from typing import Dict, List, Optional

# from pydantic import BaseModel

# from runnable import defaults, integration, utils
# from runnable.executor import BaseExecutor
# from runnable.graph import Graph
# from runnable.nodes import BaseNode

# logger = logging.getLogger(defaults.NAME)

# try:
#     from kubernetes import client
#     from kubernetes.client import V1EnvVar, V1EnvVarSource, V1PersistentVolumeClaimVolumeSource, V1SecretKeySelector
# except ImportError as _e:
#     msg = "Kubernetes Dependencies have not been installed!!"
#     # raise Exception(msg) from _e


# class Toleration(BaseModel):
#     effect: str
#     key: str
#     operator: str
#     value: str


# class K8sJobExecutor(BaseExecutor):
#     service_name = "k8s-job"

#     # TODO: move this to K8's style config.
#     class ContextConfig(BaseModel):
#         docker_image: str
#         config_path: str = ""  # Let the client decide on the path to the config file.
#         namespace: str = "default"
#         cpu_limit: str = "250m"
#         memory_limit: str = "1G"
#         gpu_limit: int = 0
#         gpu_vendor: str = "nvidia.com/gpu"
#         cpu_request: str = ""
#         memory_request: str = ""
#         active_deadline_seconds: int = 60 * 60 * 2  # 2 hours
#         ttl_seconds_after_finished: int = 60  #  1 minute
#         image_pull_policy: str = "Always"
#         secrets_from_k8s: dict = {}  # EnvVar=SecretName:Key
#         persistent_volumes: dict = {}  # volume-name:mount_path
#         node_selector: Dict[str, str] = {}
#         tolerations: List[Toleration] = []
#         labels: Dict[str, str] = {}

#     def __init__(self, config: Optional[dict] = None):
#         self.config = self.ContextConfig(**(config or {}))
#         self.persistent_volumes = {}

#         for i, (claim, mount_path) in enumerate(self.config.persistent_volumes.items()):
#             self.persistent_volumes[f"executor-{i}"] = (claim, mount_path)

#     def prepare_for_graph_execution(self):
#         """
#         This method would be called prior to calling execute_graph.
#         Perform any steps required before doing the graph execution.

#         The most common implementation is to prepare a run log for the run if the run uses local interactive compute.

#         But in cases of actual rendering the job specs (eg: AWS step functions, K8's) we need not do anything.
#         """

#         integration.validate(self, self.run_log_store)
#         integration.configure_for_traversal(self, self.run_log_store)

#         integration.validate(self, self.catalog_handler)
#         integration.configure_for_traversal(self, self.catalog_handler)

#         integration.validate(self, self.secrets_handler)
#         integration.configure_for_traversal(self, self.secrets_handler)

#         integration.validate(self, self.experiment_tracker)
#         integration.configure_for_traversal(self, self.experiment_tracker)

#     def prepare_for_node_execution(self):
#         """
#         Perform any modifications to the services prior to execution of the node.

#         Args:
#             node (Node): [description]
#             map_variable (dict, optional): [description]. Defaults to None.
#         """

#         integration.validate(self, self.run_log_store)
#         integration.configure_for_execution(self, self.run_log_store)

#         integration.validate(self, self.catalog_handler)
#         integration.configure_for_execution(self, self.catalog_handler)

#         integration.validate(self, self.secrets_handler)
#         integration.configure_for_execution(self, self.secrets_handler)

#         integration.validate(self, self.experiment_tracker)
#         integration.configure_for_execution(self, self.experiment_tracker)

#         self._set_up_run_log(exists_ok=True)

#     @property
#     def _client(self):
#         from kubernetes import config as k8s_config

#         if self.config.config_path:
#             k8s_config.load_kube_config(kube_config_path=self.config.config_path)
#         else:
#             # https://github.com/kubernetes-client/python/blob/master/kubernetes/base/config/__init__.py
#             k8s_config.load_config()
#         return client

#     @property
#     def tolerations(self):
#         return [toleration.dict() for toleration in self.config.tolerations]

#     def execute_job(self, node: BaseNode):
#         command = utils.get_job_execution_command(self, node)
#         logger.info(f"Triggering a kubernetes job with : {command}")

#         self.config.labels["job_name"] = self.run_id

#         k8s_batch = self._client.BatchV1Api()

#         cpu_limit = self.config.cpu_limit
#         memory_limit = self.config.memory_limit

#         cpu_request = self.config.cpu_request or cpu_limit
#         memory_request = self.config.memory_request or memory_limit

#         gpu_limit = str(self.config.gpu_limit)  # Should be something like nvidia -etc

#         limits = {
#             "cpu": cpu_limit,
#             "memory": memory_limit,
#             self.config.gpu_vendor: gpu_limit,
#         }
#         requests = {"cpu": cpu_request, "memory": memory_request}
#         resources = {"limits": limits, "requests": requests}

#         environment_variables = []
#         for secret_env, k8_secret in self.config.secrets_from_k8s.items():
#             try:
#                 secret_name, key = k8_secret.split(":")
#             except Exception as _e:
#                 msg = "K8's secret should be of format EnvVar=SecretName:Key"
#                 raise Exception(msg) from _e
#             secret_as_env = V1EnvVar(
#                 name=secret_env,
#                 value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(name=secret_name, key=key)),
#             )
#             environment_variables.append(secret_as_env)

#         overridden_params = utils.get_user_set_parameters()
#         # The parameters present in the environment override the parameters present in the parameters file
#         # The values are coerced to be strings, hopefully they will be fine on the other side.
#         for k, v in overridden_params.items():
#             environment_variables.append(V1EnvVar(name=defaults.PARAMETER_PREFIX + k, value=str(v)))

#         pod_volumes = []
#         volume_mounts = []
#         for claim_name, (claim, mount_path) in self.persistent_volumes.items():
#             pod_volumes.append(
#                 self._client.V1Volume(
#                     name=claim_name,
#                     persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=claim),
#                 )
#             )
#             volume_mounts.append(self._client.V1VolumeMount(name=claim_name, mount_path=mount_path))

#         base_container = self._client.V1Container(
#             name=self.run_id,
#             image=self.config.docker_image,
#             command=shlex.split(command),
#             resources=resources,
#             env=environment_variables,
#             image_pull_policy="Always",
#             volume_mounts=volume_mounts or None,
#         )

#         pod_spec = self._client.V1PodSpec(
#             volumes=pod_volumes or None,
#             restart_policy="Never",
#             containers=[base_container],
#             node_selector=self.config.node_selector,
#             tolerations=self.tolerations,
#         )

#         pod_template = self._client.V1PodTemplateSpec(
#             metadata=client.V1ObjectMeta(
#                 labels=self.config.labels,
#                 annotations={"sidecar.istio.io/inject": "false"},
#             ),
#             spec=pod_spec,
#         )

#         job_spec = client.V1JobSpec(
#             template=pod_template,
#             backoff_limit=2,
#             ttl_seconds_after_finished=self.config.ttl_seconds_after_finished,
#         )
#         job_spec.active_deadline_seconds = self.config.active_deadline_seconds

#         job = client.V1Job(
#             api_version="batch/v1",
#             kind="Job",
#             metadata=client.V1ObjectMeta(name=self.run_id),
#             spec=job_spec,
#         )

#         logger.debug(f"Submitting kubernetes job: {job}")

#         try:
#             response = k8s_batch.create_namespaced_job(
#                 body=job,
#                 namespace=self.config.namespace,
#                 _preload_content=False,
#                 pretty=True,
#             )
#             print(f"Kubernetes job {self.run_id} created")
#             logger.debug(f"Kubernetes job response: {response}")
#         except Exception as e:
#             logger.exception(e)
#             raise

#     def execute_node(self, node: BaseNode, map_variable: Optional[dict] = None, **kwargs):
#         step_log = self.run_log_store.create_step_log(node.name, node._get_step_log_name(map_variable))

#         self.add_code_identities(node=node, step_log=step_log)

#         step_log.step_type = node.node_type
#         step_log.status = defaults.PROCESSING
#         self.run_log_store.add_step_log(step_log, self.run_id)

#         super()._execute_node(node, map_variable=map_variable, **kwargs)

#         step_log = self.run_log_store.get_step_log(node._get_step_log_name(map_variable), self.run_id)
#         if step_log.status == defaults.FAIL:
#             raise Exception(f"Step {node.name} failed")

#     def execute_graph(self, dag: Graph, map_variable: Optional[dict] = None, **kwargs):
#         msg = "This executor is not supported to execute any graphs but only jobs (functions or notebooks)"
#         raise NotImplementedError(msg)

#     def send_return_code(self, stage="traversal"):
#         """
#         Convenience function used by pipeline to send return code to the caller of the cli

#         Raises:
#             Exception: If the pipeline execution failed
#         """
#         if stage != "traversal":  # traversal does no actual execution, so return code is pointless
#             run_id = self.run_id

#             run_log = self.run_log_store.get_run_log_by_id(run_id=run_id, full=False)
#             if run_log.status == defaults.FAIL:
#                 raise Exception("Pipeline execution failed")
