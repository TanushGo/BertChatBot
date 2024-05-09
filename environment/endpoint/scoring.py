import os
import torch
import subprocess
import mlflow
from pprint import pprint
import json
from transformers import BertTokenizerFast, BertForSequenceClassification, pipeline
#from optimum.bettertransformer import BetterTransformer


def init():
    global model
    global tokenizer
    global device
    global chatbot


    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    if cuda_available:
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] ID of current CUDA device: {torch.cuda.current_device()}")
        print("[INFO] nvidia-smi output:")
        pprint(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode(
                "utf-8"
            )
        )
    else:
        print(
            "[WARN] CUDA acceleration is not available. This model takes hours to run on medium size data."
        )

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")

    # load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        model_path
    )

    # Load the model
    try:
        model = BertForSequenceClassification.from_pretrained(
            model_path
        )
    except Exception as e:
        print(
            f"[ERROR] Error happened when loading the model on GPU or the default device. Error: {e}"
        )
        print("[INFO] Trying on CPU.")
        model = BertForSequenceClassification.from_pretrained(model_path)
        device = "cpu"

    # # Optimize the model
    # if device != "cpu":
    #     try:
    #         model = BetterTransformer.transform(model, keep_original_model=False)
    #         print("[INFO] BetterTransformer loaded.")
    #     except Exception as e:
    #         print(
    #             f"[ERROR] Error when converting to BetterTransformer. An unoptimized version of the model will be used.\n\t> {e}"
    #         )

    chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    mlflow.log_param("device", device)
    mlflow.log_param("model", type(model).__name__)


def predict(text):
    
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    outputs = model(**inputs)

    probs = outputs[0].softmax(1)
    pred_label_idx = probs.argmax()
    pred_label = model.config.id2label[pred_label_idx.item()]

    return {"label": pred_label, "score": probs[0][pred_label_idx.item()].item()}

def run(raw_data):

    #print(f"[INFO] Reading new mini-batch of {len(mini_batch)} file(s).")
    
    data = json.loads(raw_data)["data"]

    # reuslt = predict(data)
    result = chatbot(data)[0]
    result["id"] = model.config.label2id[result['label']]



    #mlflow.log_metric("rows_per_second", rps)
    return result
