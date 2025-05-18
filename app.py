from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

app = FastAPI()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer", local_files_only=True)

# Load TFLite models
interpreter_cls = tf.lite.Interpreter(model_path="./model_cls.tflite")
interpreter_event = tf.lite.Interpreter(model_path="./model_event.tflite")
interpreter_task = tf.lite.Interpreter(model_path="./model_task.tflite")
interpreter_device = tf.lite.Interpreter(model_path="./model_device.tflite")

# Allocate tensors
interpreter_cls.allocate_tensors()
interpreter_event.allocate_tensors()
interpreter_task.allocate_tensors()
interpreter_device.allocate_tensors()

class TextInput(BaseModel):
    text: str

def run_tflite_model(interpreter, inputs_dict):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set inputs
    for detail in input_details:
        key = detail["name"]
        if "input_ids" in key:
            interpreter.set_tensor(detail["index"], inputs_dict["input_ids"])
        elif "attention_mask" in key:
            interpreter.set_tensor(detail["index"], inputs_dict["attention_mask"])
        elif "token_type_ids" in key:
            interpreter.set_tensor(detail["index"], inputs_dict["token_type_ids"])

    # Run inference
    interpreter.invoke()

    # Get outputs
    return [interpreter.get_tensor(output["index"]) for output in output_details]

def predict_label_tflite(text):
    tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    inputs = {
        "input_ids": tokens["input_ids"].astype(np.int32),
        "attention_mask": tokens["attention_mask"].astype(np.int32),
        "token_type_ids": tokens["token_type_ids"].astype(np.int32),
    }

    outputs = run_tflite_model(interpreter_cls, inputs)
    probs = tf.nn.softmax(outputs[0], axis=-1).numpy()
    return int(np.argmax(probs))

def run_token_classification_tflite(interpreter, text):
    tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    inputs = {
        "input_ids": tokens["input_ids"].astype(np.int32),
        "attention_mask": tokens["attention_mask"].astype(np.int32),
        "token_type_ids": tokens["token_type_ids"].astype(np.int32),
    }

    outputs = run_tflite_model(interpreter, inputs)
    logits = outputs[0]  # Shape: (1, 128, num_labels)
    predictions = np.argmax(logits, axis=-1)[0]

    input_ids = inputs["input_ids"][0]
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids)

    results = []
    for token, pred in zip(tokens_list, predictions):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        results.append((token, int(pred)))
    return results

@app.post("/predict")
async def predict(input: TextInput):
    text = input.text

    label = predict_label_tflite(text)
    if label == 0:
        model_used = "EVENT"
        result = run_token_classification_tflite(interpreter_event, text)
    elif label == 1:
        model_used = "Device"
        result = run_token_classification_tflite(interpreter_device, text)
    else:
        model_used = "TASK"
        result = run_token_classification_tflite(interpreter_task, text)

    return {
        "model_used": model_used,
        "result": result
    }
