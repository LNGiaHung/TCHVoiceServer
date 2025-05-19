from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
from typing import List

app = FastAPI()

# Load tokenizer once (efficient and reused)
tokenizer = AutoTokenizer.from_pretrained("./tokenizer", local_files_only=True)

# Helper to load TFLite model on demand
def load_interpreter(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

class TextInput(BaseModel):
    text: str

def prepare_inputs(text: str):
    tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    return {
        "input_ids": tokens["input_ids"].astype(np.int32),
        "attention_mask": tokens["attention_mask"].astype(np.int32),
        "token_type_ids": tokens["token_type_ids"].astype(np.int32),
    }

def run_tflite_model(interpreter, inputs_dict):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details:
        name = detail["name"]
        if "input_ids" in name:
            interpreter.set_tensor(detail["index"], inputs_dict["input_ids"])
        elif "attention_mask" in name:
            interpreter.set_tensor(detail["index"], inputs_dict["attention_mask"])
        elif "token_type_ids" in name:
            interpreter.set_tensor(detail["index"], inputs_dict["token_type_ids"])

    interpreter.invoke()
    return [interpreter.get_tensor(out["index"]) for out in output_details]

def predict_label(text: str) -> int:
    inputs = prepare_inputs(text)
    interpreter = load_interpreter("./model_cls.tflite")
    outputs = run_tflite_model(interpreter, inputs)
    probs = tf.nn.softmax(outputs[0], axis=-1).numpy()
    return int(np.argmax(probs))

def token_classification(text: str, model_path: str) -> List:
    inputs = prepare_inputs(text)
    interpreter = load_interpreter(model_path)
    outputs = run_tflite_model(interpreter, inputs)

    logits = outputs[0]  # shape: (1, 128, num_labels)
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
    label = predict_label(text)

    if label == 0:
        model_used = "EVENT"
        result = token_classification(text, "./model_event.tflite")
    elif label == 1:
        model_used = "DEVICE"
        result = token_classification(text, "./model_device.tflite")
    else:
        model_used = "TASK"
        result = token_classification(text, "./model_task.tflite")

    return {
        "model_used": model_used,
        "result": result
    }
