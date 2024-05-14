import train
from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizerFast,BertTokenizer, TrainingArguments, Trainer
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

# Load the data asset
data_asset = ml_client.data.get(name="univeristy-info", version="uci")

# Load the data
df, label2id, id2label, num_labels = train.createDF(data_asset.path)

#Create the pretrained model with variables depending on the data
model_name = "bert-base-uncased"
max_len = 256

tokenizer = BertTokenizer.from_pretrained(model_name, 
                                          max_length=max_len)

model = BertForSequenceClassification.from_pretrained(model_name, 
                                                      num_labels=num_labels, 
                                                      id2label=id2label, 
                                                      label2id = label2id)


train_dataloader, test_dataloader = train.createDataLoader(df, tokenizer)


training_args = TrainingArguments(
    output_dir='./output', 
    do_train=True,
    do_eval=True,
    num_train_epochs=100,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=16,
    warmup_steps=100,                
    weight_decay=0.05,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',            
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps", 
    load_best_model_at_end=True
)

#Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,                 
    train_dataset=train_dataloader,         
    eval_dataset=test_dataloader,            
    compute_metrics= train.compute_metrics
)

#Run the training session
train.trainRun(trainer, model)

#evaluate the model
q=[trainer.evaluate(eval_dataset=df2) for df2 in [train_dataloader, test_dataloader]]

#pd.DataFrame(q, index=["train","test"]).iloc[:,:5]

text = "Hello"
train.predict(model, tokenizer, text)

# Save the model and tokenizer
model_path = "chatbot"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)