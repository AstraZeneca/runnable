---
title: Welcome
sidebarDepth: 0
---

# **Welcome to Magnus**
*Lets do great things together!!*

---

Magnus provides four capabilities for data teams:

- **Compute execution plan**: A DAG representation of work that you want to get done. Individual nodes of the DAG 
could be simple python or shell tasks or complex deeply nested parallel branches or embedded DAGs themselves.

- **Run log store**: A place to store run logs for reporting or re-running older runs. Along with capturing the 
status of execution,  the run logs also capture code identifiers (commits, docker image digests etc), data hashes and 
configuration settings for reproducibility and audit.

- **Data Catalogs**: A way to pass data between nodes of the graph during execution and also serves the purpose of
versioning the data used by a particular run.

- **Secrets**: A framework to provide secrets/credentials at run time to the nodes of the graph.

### Design decisions:

- **Easy to extend**: All the four capabilities are just definitions and can be implemented in many flavors.
    
    - **Compute execution plan**: You can choose to run the DAG on your local computer, in containers of local computer 
    or off load the work to cloud providers or translate the DAG to AWS step functions or Argo workflows.

    - **Run log Store**: The actual implementation of storing the run logs could be in-memory, file system, S3, 
    database etc.

    - **Data Catalogs**: The data files generated as part of a run could be stored on file-systems, S3 or could be
    extended to fit your needs.

    - **Secrets**: The secrets needed for your code to work could be in dotenv, AWS or extended to fit your needs.

- **Pipeline as contract**: Once a DAG is defined and proven to work in local or some environment, there is absolutely
no code change needed to deploy it to other environments. This enables the data teams to prove the correctness of 
the dag in dev environments while infrastructure teams to find the suitable way to deploy it.

- **Reproducibility**: Run log store and data catalogs hold the version, code commits, data files used for a run 
making it easy to re-run an older run or debug a failed run. Debug environment need not be the same as 
original environment. 

- **Easy switch**: Your infrastructure landscape changes over time. With magnus, you can switch infrastructure 
by just changing a config and not code.


Magnus does not aim to replace existing and well constructed orchestrators like AWS Step functions or 
[argo](https://argoproj.github.io/workflows/) but complements them in a unified, simple and intuitive way.