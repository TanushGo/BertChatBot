from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
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

# update the 'my_path' variable to match the location of where you downloaded the data on your
# local filesystem

my_path = "./kaggle/input/intents.json"
# set the version number of the data asset
v1 = "uci"

my_data = Data(
    name="univeristy-info",
    version=v1,
    description="Questions about college",
    path=my_path,
    type=AssetTypes.URI_FILE,
)

## create data asset if it doesn't already exist:
try:
    data_asset = ml_client.data.get(name="univeristy-info", version=v1)
    print(
        f"Data asset already exists. Name: {my_data.name}, version: {my_data.version}"
    )
except:
    ml_client.data.create_or_update(my_data)
    print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")