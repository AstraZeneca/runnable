# Dot Env

This secrets provider the environment as a secrets provider.

If a name is provided, we look for the secret in the environment. The name is case-sensitive.

If a name is not provided, we return an empty dictionary.

---
!!! Note

    Providing secrets via environment variables poses security risks. Use a secure secrets manager.

    This secrets manager returns an empty dictionary, if a name is not provided, unlike other secret managers.

---

The complete configuration

```yaml
secrets:
  type: env-secrets-manager

```
