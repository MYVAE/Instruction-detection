import torch
import argparse
import numpy as np
import joblib
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tqdm import tqdm
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradient test")
    parser.add_argument("--start", type=str, default=0,
                       help="Start index")
    parser.add_argument("--end", type=int, default=33,
                       help="End index")
    return parser.parse_args()

args = parse_args()
model_name = "meta-llama/Llama-3.1-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)

def process_prompts_instruct(prompt):
    chat = [{"role": "user", "content": f"{prompt}"}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

def get_hidden_states(data):
    inputs = tokenizer(process_prompts_instruct(data), return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    last_token_hidden_states = []
    for layer_hidden_state in hidden_states:
        last_token_hidden_state = layer_hidden_state[:, -1, :].cpu().numpy()
        last_token_hidden_states.append(last_token_hidden_state.squeeze())
    # print(np.array(last_token_hidden_states).shape)
    return np.array(last_token_hidden_states)



# Prepare features
instructions_train = get_instruction_1_train()
data_train = get_data_wiki_train()
mixed_data_train = []
for i, context in enumerate(data_train):
    attack = instructions_train[i % len(instructions_train)]
    mixed_data_train.append(insert_middle(context, attack))
data_embeddings_train = [get_hidden_states(data) for data in tqdm(data_train, desc="Processing data_train")]
with open('hidden_states/train_data.pkl', 'wb') as f:
    pickle.dump(data_embeddings_train, f)
instruction_embeddings_train = [get_hidden_states(data) for data in tqdm(mixed_data_train, desc="Processing mixed_data_train")]
with open('hidden_states/train_instruction.pkl', 'wb') as f:
    pickle.dump(instruction_embeddings_train, f)
x_train = np.array(data_embeddings_train + instruction_embeddings_train)
y_train = np.array([0] * len(data_embeddings_train) + [1] * len(instruction_embeddings_train))
x_train, y_train = shuffle(x_train, y_train, random_state=42)


instructions_test = get_instruction_1_test()
data_test = get_data_wiki_test()
mixed_data_test = []
for i, context in enumerate(data_test):
    attack = instructions_test[i % len(instructions_test)]
    mixed_data_test.append(insert_middle(context, attack))
data_embeddings_test = [get_hidden_states(data) for data in tqdm(data_test, desc="Processing data_test")]
with open('hidden_states/test_data1.pkl', 'wb') as f:
    pickle.dump(data_embeddings_test, f)
instruction_embeddings_test = [get_hidden_states(data) for data in tqdm(mixed_data_test, desc="Processing mixed_data_test")]
with open('hidden_states/test_instruction1.pkl', 'wb') as f:
    pickle.dump(instruction_embeddings_test, f)
x_test_1 = np.array(data_embeddings_test + instruction_embeddings_test)
y_test_1 = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))


data_test = get_data_news_test()
mixed_data_test = []
for i, context in enumerate(data_test):
    attack = instructions_test[i % len(instructions_test)]
    mixed_data_test.append(insert_middle(context, attack))
data_embeddings_test = [get_hidden_states(data) for data in tqdm(data_test, desc="Processing data_test")]
with open('hidden_states/test_data2.pkl', 'wb') as f:
    pickle.dump(data_embeddings_test, f)
instruction_embeddings_test = [get_hidden_states(data) for data in tqdm(mixed_data_test, desc="Processing mixed_data_test")]
with open('hidden_states/test_instruction2.pkl', 'wb') as f:
    pickle.dump(instruction_embeddings_test, f)
x_test_2 = np.array(data_embeddings_test + instruction_embeddings_test)
y_test_2 = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))

instructions_test = get_instruction_2_test()
data_test = get_data_wiki_test()
mixed_data_test = []
for i, context in enumerate(data_test):
    attack = instructions_test[i % len(instructions_test)]
    mixed_data_test.append(insert_middle(context, attack))
data_embeddings_test = [get_hidden_states(data) for data in tqdm(data_test, desc="Processing data_test")]
with open('hidden_states/test_data3.pkl', 'wb') as f:
    pickle.dump(data_embeddings_test, f)
instruction_embeddings_test = [get_hidden_states(data) for data in tqdm(mixed_data_test, desc="Processing mixed_data_test")]
with open('hidden_states/test_instruction3.pkl', 'wb') as f:
    pickle.dump(instruction_embeddings_test, f)
x_test_3 = np.array(data_embeddings_test + instruction_embeddings_test)
y_test_3 = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))

data_test = get_data_news_test()
mixed_data_test = []
for i, context in enumerate(data_test):
    attack = instructions_test[i % len(instructions_test)]
    mixed_data_test.append(insert_middle(context, attack))
data_embeddings_test = [get_hidden_states(data) for data in tqdm(data_test, desc="Processing data_test")]
with open('hidden_states/test_data4.pkl', 'wb') as f:
    pickle.dump(data_embeddings_test, f)
instruction_embeddings_test = [get_hidden_states(data) for data in tqdm(mixed_data_test, desc="Processing mixed_data_test")]
with open('hidden_states/test_instruction4.pkl', 'wb') as f:
    pickle.dump(instruction_embeddings_test, f)
x_test_4 = np.array(data_embeddings_test + instruction_embeddings_test)
y_test_4 = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))


layer_accuracies1 = {}
layer_accuracies2 = {}
layer_accuracies3 = {}
layer_accuracies4 = {}
for layer_index in range(args.start, args.end):
    # Train
    x_train_layer = x_train[:, layer_index, :]
    mlp = MLPClassifier(hidden_layer_sizes=(1024, 256, 64, 16), max_iter=5000, verbose=True, n_iter_no_change=10)
    mlp.fit(x_train_layer, y_train)
    joblib.dump(mlp, 'hidden_state.pkl')

    # Test
    x_test_layer_1 = x_test_1[:, layer_index, :]
    y_pred_1 = mlp.predict(x_test_layer_1)
    accuracy = accuracy_score(y_test_1, y_pred_1)
    layer_accuracies1[layer_index] = accuracy

    x_test_layer_2 = x_test_2[:, layer_index, :]
    y_pred_2 = mlp.predict(x_test_layer_2)
    accuracy = accuracy_score(y_test_2, y_pred_2)
    layer_accuracies2[layer_index] = accuracy

    x_test_layer_3 = x_test_3[:, layer_index, :]
    y_pred_3 = mlp.predict(x_test_layer_3)
    accuracy = accuracy_score(y_test_3, y_pred_3)
    layer_accuracies3[layer_index] = accuracy

    x_test_layer_4 = x_test_4[:, layer_index, :]
    y_pred_4 = mlp.predict(x_test_layer_4)
    accuracy = accuracy_score(y_test_4, y_pred_4)
    layer_accuracies4[layer_index] = accuracy

for layer, accuracy in layer_accuracies1.items():
    print(f"Test Accuracy1 for Layer {layer}: {accuracy * 100:.2f}%")
for layer, accuracy in layer_accuracies2.items():
    print(f"Test Accuracy2 for Layer {layer}: {accuracy * 100:.2f}%")
for layer, accuracy in layer_accuracies3.items():
    print(f"Test Accuracy3 for Layer {layer}: {accuracy * 100:.2f}%")
for layer, accuracy in layer_accuracies4.items():
    print(f"Test Accuracy4 for Layer {layer}: {accuracy * 100:.2f}%")
