import uuid

# Create a unique name for the endpoint
online_endpoint_name = "chatbot-endpoint-" + str(uuid.uuid4())[:8]

from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="65a7bb82-b5cd-4770-a41d-be2cf0a96be8",
    resource_group_name="DVSM",
    workspace_name="BertChatBot",
)


# define an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Endpoint for Bert Chatbot",
    auth_mode="key",
    tags={
        "training_dataset": "university-info",
    },
)

# create the online endpoint
# expect the endpoint to take approximately 2 minutes.

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

print(
    f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
)