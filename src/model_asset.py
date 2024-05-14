from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizerFast

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

model_path = "chatbot"

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)

# Create a pipeline
chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

model_path = "chatbot_pipelines"
chatbot.save_pretrained(model_path)

# Register the model
model_name = 'BertChatBotPipeline'
model = ml_client.models.create_or_update(
    Model(name=model_name, path='chatbot_pipelines', type=AssetTypes.CUSTOM_MODEL)
)