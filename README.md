# BERT ChatBot
--------------------------------------------------------------

![](https://github.com/TanushGo/BertChatBot/blob/main/assets/Github_video.gif)

Website: <https://bertchatbot.azurewebsites.net/>
// Currently not live

Overview
Sentiment Analysis with Bidirectional Encoding Representation Transforms using the Hugging Face library. The chatbot uses the knowledge base to answer questions to query people may have. The chatbot is knowledgable about questions relating to university and provides students with information about being a student at the University of California, Irvine.

The jupyter file explains a lot about the process of using gradio, developing the model with the dataset, and the various different aspects of Microsoft Azure. 

## Model Development & Training
The model is trained using the data with tokenizer to better fit to the project scope. It was developed on the basis of accurately prediciting tags that a question could belong to. This simplifies the problem of question answering to utilizing a list of answers instead of generating new answers which will require a much larger initial dataset along with greater compute resources. This also solves the problem scope of having a chat application specifically for a platform like a school website or an informational booth. 

The model is trained on the dataset, and has an evaluation accuracy of around 92% it successfully creates question labels for a large variety of questions. This increases the usablility in a commercial setting. 
![image](https://github.com/TanushGo/BertChatBot/assets/94217537/5d5cbf2f-5e83-418a-8e50-41db52345f61)

## Azure 
![DVSM](https://github.com/TanushGo/BertChatBot/assets/94217537/6c933bc2-5666-497b-beaa-0d588630972f)

### Environment
Azure uses docker images with conda to develop the environment in which the model is run. The conda.yaml lists the various dependencies used by the model and in the development processThe conda.yaml file in the environment folder



### Azure Model Endpoint


### App Deployment
The deployment is made on a Azure web app hosted in the cloud. It calls the model endpoint with query inputs from the user and gives out results. The deployment is done with python using the requirments.txt and the main.py runs on the server to create the frontend with gradio and connect it with REST api to the model. The frontend deployment uses FASTapi for creating the app. 

`python -m gunicorn main:app -k uvicorn.workers.UvicornWorker` 

is the command run on deployment to create the app and handle the api calls from the backend as well as with we browsers. 
The app is currently offline to preserve cost. 
