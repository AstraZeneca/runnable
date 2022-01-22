# Database

This Run Log store stores the run logs in a database in concurrent compliant ways.
The Run Log is split into several small steps that are individually concurrent safe.

You can retrieve the complete run log too when needed.

When to use:

- When you want to compare logs between runs.
- During testing in cloud/local environments.
- For production grade runs in cloud environments.
- When you have parallel branches as part of your pipeline definition.
- All compute modes accept this as a Run Log store.



## Configuration

The configuration is as follows:

```yaml
run_log:
  type: db
  config:
    connection_string:
```

### connection_string

The connection string of the database connection which is SQLAlchemy compliant.

The connection string can contain placeholders which are replaced by values from secrets at run-time.

Example:
```yaml
run_log:
  type: db
  config:
    connection_string: postgresql://${username}:${password}@${dbhost}/${dbname}
```

Would be resolved to an appropriate connection string if secrets have values for ```username```, ```password```,
```dbhost``` and ```dbname```.

## Creating the table

Magnus needs a particular table structure to store the logs in the database.

You could create it by running the following script:

```python
from magnus.datastore_extensions import db
connection_string = ...
db.create_tables(connection_string)
```
