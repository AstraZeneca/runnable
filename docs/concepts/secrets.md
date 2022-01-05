# Overview

Secrets are essential in making your data science projects secure and collaborative. They could be database credentials, API keys or any information that need to present at the run-time but invisible at all other times.Magnus provides a clean interface to access/store secrets and independent of the actual secret provider, the interface remains the same. 

As with all modules of magnus, there are many secrets providers and if none fit your needs, it is easier to write one of your to fit your needs. In magnus, all secrets are key value pairs. 

## Configuration

Configuration of a Secrets is as follows:

```yaml
secrets:
  type: 
  config:
```

### type

The type of secrets provider you want. This should be one of the secrets types already available. 

There is no default secrets provider.

### config

Any configuration parameters the secret provider accepts. 


## Interaction with Run Log Store

Magnus allows Run Log store to have secrets as placeholders for a few providers. Please refer to the Run Log store 
providers for the format.

## Imteraction within code

Secrets is the only implementation that requires you to ```import magnus``` in the code to access secrets.
This is mostly to follow the best safety guidelines though we are actively investigating options. 

Once a secret configuration is defined as above, you can access the secret by using ```get_secret``` of magnus. 
If a key is providd to the API, we return only the value associated with the secret by the key. 
If a key is not provided, we return all the key value secret pairs provided. 
The API would raise an exception if a secret by the key requested does not exist.

Currently, there is no providsion to update/edit secrets via code. 


For example if the secret key-value pairs are:

```yaml
secret_answer: 42
secret_question: everything
```

And for the code:
```python
# In my_module.py
from magnus import get_secret

def my_cool_function():

    secret = get_secret('secret_answer')

    all_secrets = get_secret()

```

secret would have a value of ```42``` while all_secrets would be a dictionary ```{'secret_answer': 42, 'secret_question': 'everything'}```


## Parameterised definition

As with any part of the mangus configuration, you can parameterise the configuration of secrets to switch between 
providers without changing the base definition. 

Please follow the example provided [here](../dag/#parameterized_definition) for more information. 

### TODO

- [ ] Expose secrets as ENV variables with a known prefix and remove them after function call


## Extensions

You can easily extend magnus to bring in your custom provider, if a default
implementation does not exist or you are not happy with the implementation. 
