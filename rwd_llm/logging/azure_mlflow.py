# support MLFlow logging to Azure Machine Learning Workspace.
# Note that for this to work, the azure-mlflow package must be installed
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-cli-runs
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking

from typing import Any


def get_ml_client(
    credential: Any,  # actually azure.identity.Credential
    azure_subscription_id: str,
    aml_workspace_resource_group: str,
    aml_workspace_name: str,
):
    from azure.ai.ml import MLClient

    # Enter details of your AzureML workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=azure_subscription_id,
        resource_group_name=aml_workspace_resource_group,
        workspace_name=aml_workspace_name,
    )
    return ml_client


def get_aml_mlflow_tracking_uri_from_credential(
    credential: Any,
    azure_subscription_id: str,
    aml_workspace_resource_group: str,
    aml_workspace_name: str,
):
    ml_client = get_ml_client(
        credential=credential,
        azure_subscription_id=azure_subscription_id,
        aml_workspace_resource_group=aml_workspace_resource_group,
        aml_workspace_name=aml_workspace_name,
    )
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    return mlflow_tracking_uri


def get_aml_mlflow_tracking_uri(
    aml_workspace_region: str,
    azure_subscription_id: str,
    aml_workspace_resource_group: str,
    aml_workspace_name: str,
):
    # try to directly configure the URI
    # see 'manual' instructions in:
    # https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking
    mlflow_tracking_uri = (
        f"azureml://{aml_workspace_region}.api.azureml.ms/mlflow/v1.0/"
        f"subscriptions/{azure_subscription_id}/"
        f"resourceGroups/{aml_workspace_resource_group}/"
        "providers/Microsoft.MachineLearningServices/"
        f"workspaces/{aml_workspace_name}"
    )
    return mlflow_tracking_uri
