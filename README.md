# BERT ChatBot
--------------------------------------------------------------

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
![Azure](https://img.shields.io/badge/azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![YAML](https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Gunicorn](https://img.shields.io/badge/gunicorn-%298729.svg?style=for-the-badge&logo=gunicorn&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)



![](https://github.com/TanushGo/BertChatBot/blob/main/assets/Github_video.gif)

Website: <https://bertchatbot.azurewebsites.net/>
// Currently not live

Overview
Sentiment Analysis with Bidirectional Encoding Representation Transforms using the Hugging Face library. The chatbot uses the knowledge base to answer questions to query people may have. The chatbot is knowledgable about questions relating to university and provides students with information about being a student at the University of California, Irvine.

The jupyter file explains a lot about the process of using gradio, developing the model with the dataset, and the various different aspects of Microsoft Azure. 

## Model Development & Training :notebook_with_decorative_cover:
The model is trained using the data with tokenizer to better fit to the project scope. It was developed on the basis of accurately prediciting tags that a question could belong to. This simplifies the problem of question answering to utilizing a list of answers instead of generating new answers which will require a much larger initial dataset along with greater compute resources. This also solves the problem scope of having a chat application specifically for a platform like a school website or an informational booth. 

The model is trained on the dataset, and has an evaluation accuracy of around 92% it successfully creates question labels for a large variety of questions. This increases the usablility in a commercial setting. 


![image](https://github.com/TanushGo/BertChatBot/assets/94217537/5d5cbf2f-5e83-418a-8e50-41db52345f61)

## Azure :computer:
Azure is a great platform to train and deploy models developed using python libraries. There is a bit of a learning curve in using the platform and sdk. The documentation and forums are good place to start. The app utilizes an Azure ML workspace for training and deploying the model. The Azure Web App is used to connect with the model endpoint so that the trained model can make predictions from user input. 


![DVSM](https://github.com/TanushGo/BertChatBot/assets/94217537/6c933bc2-5666-497b-beaa-0d588630972f)

### Environment
Azure uses docker images with conda to develop the environment in which the model is run. The conda.yaml lists the various dependencies used by the model and in the development process. The docker image uses an ubuntu deployment and conda downloads all the dependices. The conda.yaml file is in the environment folder



### Azure Model Endpoint
The scoring file takes the data provided by the api call, and loads the model on the endpoint deployment. The model endpoint runs the deployment of the model on the text and returns the tag prediction utilizing the api. The endpoint utilizes the key to protect the endpoint and encrpyt the data. 


### App Deployment :satellite:
The deployment is made on a Azure web app hosted in the cloud. It calls the model endpoint with query inputs from the user and gives out results. The deployment is done with python using the requirments.txt and the main.py runs on the server to create the frontend with gradio and connect it with REST api to the model. The frontend deployment uses FASTapi for creating the app. 

`python -m gunicorn main:app -k uvicorn.workers.UvicornWorker` 

is the command run on deployment to create the app and handle the api calls from the backend as well as with we browsers. 
The app is currently offline to preserve cost. 
