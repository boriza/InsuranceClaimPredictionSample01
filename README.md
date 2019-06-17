## InsuranceClaimPredictionSample01


Azure ML CLI Documentation

https://docs.microsoft.com/en-us/azure/machine-learning/service/reference-azure-machine-learning-cli


Attach Azure ML to the current directory. Execute this in root of the project 
```
az ml folder attach -w aml -g databricks-au

az ml folder attach [--experiment-name]
                    [--output-metadata-file]
                    [--path]
                    [--resource-group]
                    [--subscription-id]
                    [--workspace-name]


```

Get a list of available expriments in your AML workspace
```
az ml experiment list -w aml

az ml experiment list [--output-metadata-file]
                      [--path]
                      [--resource-group]
                      [--subscription-id]
                      [--workspace-name]

```
Submit current code for execusion in Azure

```
az ml run submit-script -c sklearn -e cli-test

az ml run submit-script [--async]
                        [--conda-dependencies]
                        [--ct]
                        [--experiment-name]
                        [--output-metadata-file]
                        [--path]
                        [--resource-group]
                        [--run-configuration-name]
                        [--subscription-id]
                        [--workspace-name]
                        []
--experiment-name -e
Experiment name.

--run-configuration-name -c
Path to a run configuration file inside the given source folder.

--conda-dependencies -d
Override the default Conda dependencies file.

-ct 
compute target

--debug


```
