from sklearn.model_selection import train_test_split
import Dataloader
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import torch

#Compute the metrics for the model
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

#Create a dataframe from the json file
def createDF(file):
    #Load Data from the file
    intents = pd.read_json(file)

    #Create a panda dataframe from the data
    df = Dataloader.create_df()
    df = Dataloader.extract_json_info(intents, df)

    #Create a list of labbels used for classifying questions
    labels = df['Tag'].unique().tolist()
    labels = [s.strip() for s in labels]

    #Connecting tags with labels
    num_labels = len(labels)
    id2label = {id:label for id, label in enumerate(labels)}
    label2id = {label:id for id, label in enumerate(labels)}

    df['labels'] = df['Tag'].map(lambda x: label2id[x.strip()])

    return df, label2id, id2label, num_labels

#Create a dataloader for the model
def createDataLoader(df, tokenizer):
    #Getting the data from the frame and splitting into training and testing data
    X, y= list(df['Pattern']), list(df['labels'])
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 123)

    train_encoding = tokenizer(X_train, truncation=True, padding=True)
    test_encoding = tokenizer(X_test, truncation=True, padding=True)

    train_dataloader = Dataloader.DataLoader(train_encoding, y_train)
    test_dataloader = Dataloader.DataLoader(test_encoding, y_test)

    return train_dataloader, test_dataloader

#Train the model
def trainRun(trainer,model):
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    trainer.train()

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="BertChatBot_mlflow",
        artifact_path="BertChatBot_mlflow",
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path="BertChatBot/trained_model",
    )
    ###########################
    #</save and register model>
    ###########################

    # Stop Logging
    mlflow.end_run()

#Predict the label of the text
def predict(model, tokenizer, text):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    outputs = model(**inputs)

    probs = outputs[0].softmax(1)
    pred_label_idx = probs.argmax()
    pred_label = model.config.id2label[pred_label_idx.item()]

    #return probs, pred_label_idx, pred_label
    return {"label": pred_label, "score": probs[0][pred_label_idx.item()].item() }