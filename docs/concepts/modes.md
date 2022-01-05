# Compute Modes

In magnus, a compute mode controls the way a dag is interpreted. In some modes, we do the actual execution
of the dag while in some modes it only renders a dag definition language for a specific orchestrator. 

Conceptually, a mode can be one of two types:

- **Interactive mode**: In this mode, the dag definition is actually executed by magnus and usually it is invoked as

```shell
magnus execute --file my-project.yaml --var-file variables.yaml
```

Magnus takes care of traversal of the dag and execution of the graph in the compute you requested. Examples
of this mode or local, local container, local aws batch etc.

- **Orchestration mode**: In this mode, the dag definition is translated to your preferred orchestration language
of dag definition. To still achieve the capabilities of interactive mode, the orchestration languge is
directed to call an internal method instead of your actual function.

Specifically, the orchestration is asked to call

```shell
mangus execute_single_node --file my-project.yaml --var-file variables.yaml --step-name step-to-call
```

The branches of the original dag are also translated to the orchestrators language if its supported. If the
orchestration mode does not support a feature that magnus supports, you could still make it work by a mixed model.

Examples of orchestration modes are aws step functions, kubeflow job specification, argo job specification etc. 

## Configuration

As with any system within magnus, configuration of a mode can be done by:

```yaml
mode:
  type: 
  config:
```

### type

The type of mode provider you want. This should be one of the mode types already available. 

Local mode is provided by default if nothing is provided. 

### config

Any configuration parameters the mode provider accepts. 

## Parameterised definition

As with any part of the mangus configuration, you can parameterise the configuration of Mode to switch between 
Mode providers without changing the base definition. 

Please follow the example provided [here](../dag/#parameterized_definition) for more information. 

## Extensions

You can easily extend magnus to interpret the dag definition to a orchestration language of choice, if a default
implementation does not exist or you are not happy with the implementation. 
