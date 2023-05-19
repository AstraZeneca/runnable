# AWS Secrets manager

This package is an extension to [magnus](https://github.com/AstraZeneca/magnus-core).

## Provides

Provides functionality to use AWS secrets manager to provide secrets

## Installation instructions

```pip install magnus_extension_secrets_aws_secrets_manager```

## Set up required to use the extension

Access to AWS environment either via:

- AWS profile, generally stored in ~/.aws/credentials
- AWS credentials available as environment variables

If you are using environmental variables for AWS credentials, please set:

- AWS_ACCESS_KEY_ID: AWS access key
- AWS_SECRET_ACCESS_KEY: AWS access secret
- AWS_SESSION_TOKEN: The session token, useful to assume other roles

A AWS secrets store that you want to use to store the the secrets.

## Config parameters

The full configuration of the AWS secrets manager is:

```yaml
secrets:
  type: 'aws-secrets-manager'
  config:
    secret_arn: The secret ARN to retrieve the secrets from.
    region: # Region if we are using
    aws_profile: #The profile to use or default
    use_credentials: # Defaults to False
```

### **secret_arn**:

The arn of the secret that you want to use. Internally, we use boto3 to access the secrets.

The below parameters are inherited from AWS Configuration.

### **aws_profile**:

The profile to use for acquiring boto3 sessions.

Defaults to None, which is used if its role based access or in case of credentials present as environmental variables.

### **region**:

The region to use for acquiring boto3 sessions.

Defaults to *eu-west-1*.


### **aws_credentials_file**:

The file containing the aws credentials.

Defaults to ```~/.aws/credentials```.

### **use_credentials**:

Set it to ```True``` to provide AWS credentials via environmental variables.

Defaults to ```False```.

### ***role_arn***:

The role to assume after getting the boto3 session.

**This is required if you are using ```use_credentials```.**
