from sklearn.model_selection import train_test_split
import train
from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizerFast,BertTokenizer, TrainingArguments, Trainer



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

file = "kaggle\input\intents.json"
df, label2id, id2label, num_labels = train.createDF(file)

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

trainer = Trainer(
    model=model,
    args=training_args,                 
    train_dataset=train_dataloader,         
    eval_dataset=test_dataloader,            
    compute_metrics= train.compute_metrics
)

q=[trainer.evaluate(eval_dataset=df2) for df2 in [train_dataloader, test_dataloader]]

#pd.DataFrame(q, index=["train","test"]).iloc[:,:5]

text = "Hello"
train.predict(model, tokenizer, text)