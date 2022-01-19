# Do nothing catalog provider

Use this catalog provider if you do not want to use the cataloging functionality.

The complete configuration:
```yaml
catalog:
  type: do-nothing
  
dag:
  ...
```

The individual steps could have ```get``` and ```put``` phases but since the catalog handler does nothing, these files
would not be cataloged.


## Design thought

Use this catalog type to temporarily switch of cataloging in local mode for debugging purposes.