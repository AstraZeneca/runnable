# Dot Env

This secrets provider uses a file to store secrets. The naming convention for such file is the ```.env``` file.


---
!!! Note

    This secrets provider should only be used for local modes and for development purpose only.

    Please be sure on **NOT** committing these files to your git and if possible, add them to .gitignore.

---

The complete configuration

```yaml
secrets:
  type: dotenv
  config:
    location:

```

## location

The location of the file from which the secrets should be loaded.

Defaults to .env file in the project root directory.

## Format

The format of contents of the secrets file should be

```shell
secret_name=secret_value#Any comment that you want to pass
```

Any content after ```#``` is considered a comment and ignored.

A exception would be raised if the secret naming does not follow these standards.
