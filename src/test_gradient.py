import torch
import argparse
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tqdm import tqdm
import torch.nn.functional as F
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradient test")
    parser.add_argument("--start", type=str, default=0,
                       help="Start index")
    parser.add_argument("--end", type=int, default=32,
                       help="End index")
    return parser.parse_args()

args = parse_args()
model_name = "meta-llama/Llama-3.1-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)


def get_input_gradient(input_text, target_text="Sure"):
    target_shapes = {
        'self_attn.q_proj.weight': (400, 400),
        'self_attn.k_proj.weight': (100, 400),
        'self_attn.v_proj.weight': (100, 400),
        'self_attn.o_proj.weight': (400, 400),
    }
    layer_gradients = []

    prompt = (
        f'<s>[INST] <<SYS>> {{system_prompt}} <</SYS>> {{content}} [/INST]' + 
        f'{{sep_token}} {{summary}} {{eos_token}}'
    )
    def apply_prompt_template(sample):
        txt = prompt.format(
            system_prompt='You are a helpful assistant. Help me with the following query: ',
            content=sample['source'],
            summary=sample['target'],
            eos_token=tokenizer.eos_token,
            sep_token=tokenizer.eos_token,
        )
        return {
            'text': txt,
        }     
    
    basic_sample = {"source": input_text, "target": target_text}
    d = apply_prompt_template(basic_sample)
    
    input_ids = tokenizer(d['text']).input_ids
    sep = [i for i, token_id in enumerate(input_ids) if token_id == tokenizer.eos_token_id][-2]
    input_ids = input_ids[:sep] + input_ids[sep+1:]
    input_ids = torch.tensor(np.array([input_ids])).to(model.device)
    
    target_ids = input_ids.clone()
    target_ids[:, :sep] = -100
    
    model.zero_grad()
    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss
    neg_log_likelihood.backward()

    del input_ids
    del target_ids
    del d
    del basic_sample
    del outputs
    del neg_log_likelihood

    def apply_cnn_reduction(gradient, target_shape):
        input_tensor = torch.tensor(gradient).unsqueeze(0).unsqueeze(0)
        output_tensor = F.adaptive_max_pool2d(input_tensor, target_shape)
        return output_tensor.squeeze(0).squeeze(0).numpy()
    
    for name, param in model.named_parameters():
        for key, target_shape in target_shapes.items():
            if key in name:
                grad = param.grad.detach().cpu().numpy()
                reduced_grad = apply_cnn_reduction(grad, target_shape)
                layer_gradients.append(reduced_grad)
                break

    torch.cuda.empty_cache()
    result_vectors = []
    for i in range(0, len(layer_gradients), 4):
        q_proj, k_proj, v_proj, o_proj = layer_gradients[i:i+4]
        concatenated = np.vstack([q_proj, k_proj, v_proj, o_proj])
        flattened = concatenated.flatten()
        result_vectors.append(flattened)
    return np.array(result_vectors)


# Prepare features
instructions_train = get_instruction_1_train()
data_train = get_data_wiki_train()
mixed_data_train = []
for i, context in enumerate(data_train):
    attack = instructions_train[i % len(instructions_train)]
    mixed_data_train.append(insert_middle(context, attack))
data_embeddings_train = np.array([get_input_gradient(data) for data in tqdm(data_train, desc="Processing data_train")])
for layer_index in range(data_embeddings_train.shape[1]): 
    with open(f'gradients/train_data_{layer_index}.pkl', 'wb') as f:
        pickle.dump(data_embeddings_train[:,layer_index,:], f)
del data_embeddings_train
instruction_embeddings_train = np.array([get_input_gradient(data) for data in tqdm(mixed_data_train, desc="Processing mixed_data_train")])
for layer_index in range(instruction_embeddings_train.shape[1]):
    with open(f'gradients/train_instruction_{layer_index}.pkl', 'wb') as f:
        pickle.dump(instruction_embeddings_train[:, layer_index, :], f)
del instruction_embeddings_train

instructions_test = get_instruction_1_test()
data_test = get_data_wiki_test()
mixed_data_test = []
for i, context in enumerate(data_test):
    attack = instructions_test[i % len(instructions_test)]
    mixed_data_test.append(insert_middle(context, attack))
data_embeddings_test = np.array([get_input_gradient(data) for data in tqdm(data_test, desc="Processing data_test")])
for layer_index in range(data_embeddings_test.shape[1]):
    with open(f'gradients/test_data1_{layer_index}.pkl', 'wb') as f:
        pickle.dump(data_embeddings_test[:, layer_index, :], f)
del data_embeddings_test
instruction_embeddings_test = np.array([get_input_gradient(data) for data in tqdm(mixed_data_test, desc="Processing mixed_data_test")])
for layer_index in range(instruction_embeddings_test.shape[1]):
    with open(f'gradients/test_instruction1_{layer_index}.pkl', 'wb') as f:
        pickle.dump(instruction_embeddings_test[:, layer_index, :], f)
del instruction_embeddings_test

data_test = get_data_news_test()
mixed_data_test = []
for i, context in enumerate(data_test):
    attack = instructions_test[i % len(instructions_test)]
    mixed_data_test.append(insert_middle(context, attack))
data_embeddings_test = np.array([get_input_gradient(data) for data in tqdm(data_test, desc="Processing data_test")])
for layer_index in range(data_embeddings_test.shape[1]):
    with open(f'gradients/test_data2_{layer_index}.pkl', 'wb') as f:
        pickle.dump(data_embeddings_test[:, layer_index, :], f)
del data_embeddings_test
instruction_embeddings_test = np.array([get_input_gradient(data) for data in tqdm(mixed_data_test, desc="Processing mixed_data_test")])
for layer_index in range(instruction_embeddings_test.shape[1]):
    with open(f'gradients/test_instruction2_{layer_index}.pkl', 'wb') as f:
        pickle.dump(instruction_embeddings_test[:, layer_index, :], f)
del instruction_embeddings_test

instructions_test = get_instruction_2_test()
data_test = get_data_wiki_test()
mixed_data_test = []
for i, context in enumerate(data_test):
    attack = instructions_test[i % len(instructions_test)]
    mixed_data_test.append(insert_middle(context, attack))
data_embeddings_test = np.array([get_input_gradient(data) for data in tqdm(data_test, desc="Processing data_test")])
for layer_index in range(data_embeddings_test.shape[1]):
    with open(f'gradients/test_data3_{layer_index}.pkl', 'wb') as f:
        pickle.dump(data_embeddings_test[:, layer_index, :], f)
del data_embeddings_test
instruction_embeddings_test = np.array([get_input_gradient(data) for data in tqdm(mixed_data_test, desc="Processing mixed_data_test")])
for layer_index in range(instruction_embeddings_test.shape[1]):
    with open(f'gradients/test_instruction3_{layer_index}.pkl', 'wb') as f:
        pickle.dump(instruction_embeddings_test[:, layer_index, :], f)
del instruction_embeddings_test


data_test = get_data_news_test()
mixed_data_test = []
for i, context in enumerate(data_test):
    attack = instructions_test[i % len(instructions_test)]
    mixed_data_test.append(insert_middle(context, attack))
data_embeddings_test = np.array([get_input_gradient(data) for data in tqdm(data_test, desc="Processing data_test")])
for layer_index in range(data_embeddings_test.shape[1]):
    with open(f'gradients/test_data4_{layer_index}.pkl', 'wb') as f:
        pickle.dump(data_embeddings_test[:, layer_index, :], f)
del data_embeddings_test
instruction_embeddings_test = np.array([get_input_gradient(data) for data in tqdm(mixed_data_test, desc="Processing mixed_data_test")])
for layer_index in range(instruction_embeddings_test.shape[1]):
    with open(f'gradients/test_instruction4_{layer_index}.pkl', 'wb') as f:
        pickle.dump(instruction_embeddings_test[:, layer_index, :], f)
del instruction_embeddings_test


layer_accuracies1 = {}
layer_accuracies2 = {}
layer_accuracies3 = {}
layer_accuracies4 = {}
for layer_index in range(args.start, args.end):
    # Train
    with open(f'gradients/train_data1_{layer_index}.pkl', 'rb') as f:
        data_embeddings_train = pickle.load(f)
    with open(f'gradients/train_instruction1_{layer_index}.pkl', 'rb') as f:
        instruction_embeddings_train = pickle.load(f)
    x_train_layer = np.concatenate([data_embeddings_train, instruction_embeddings_train], axis=0)
    y_train = np.array([0] * len(data_embeddings_train) + [1] * len(instruction_embeddings_train))
    x_train_layer, y_train = shuffle(x_train_layer, y_train, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(1024, 256, 64, 16), max_iter=5000, verbose=False, n_iter_no_change=50)
    mlp.fit(x_train_layer, y_train)

    # Test
    with open(f'gradients/test_data1_{layer_index}.pkl', 'rb') as f:
        data_embeddings_test = pickle.load(f)
    with open(f'gradients/test_instruction1_{layer_index}.pkl', 'rb') as f:
        instruction_embeddings_test = pickle.load(f)
    x_test_layer = np.concatenate([data_embeddings_test, instruction_embeddings_test], axis=0)
    y_test = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))
    x_test_layer, y_test = shuffle(x_test_layer, y_test, random_state=42)
    y_pred = mlp.predict(x_test_layer)
    accuracy = accuracy_score(y_test, y_pred)
    layer_accuracies1[layer_index] = accuracy

    with open(f'gradients/test_data2_{layer_index}.pkl', 'rb') as f:
        data_embeddings_test = pickle.load(f)
    with open(f'gradients/test_instruction2_{layer_index}.pkl', 'rb') as f:
        instruction_embeddings_test = pickle.load(f)
    x_test_layer = np.concatenate([data_embeddings_test, instruction_embeddings_test], axis=0)
    y_test = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))
    x_test_layer, y_test = shuffle(x_test_layer, y_test, random_state=42)
    y_pred = mlp.predict(x_test_layer)
    accuracy = accuracy_score(y_test, y_pred)
    layer_accuracies2[layer_index] = accuracy

    with open(f'gradients/test_data3_{layer_index}.pkl', 'rb') as f:
        data_embeddings_test = pickle.load(f)
    with open(f'gradients/test_instruction3_{layer_index}.pkl', 'rb') as f:
        instruction_embeddings_test = pickle.load(f)
    x_test_layer = np.concatenate([data_embeddings_test, instruction_embeddings_test], axis=0)
    y_test = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))
    x_test_layer, y_test = shuffle(x_test_layer, y_test, random_state=42)
    y_pred = mlp.predict(x_test_layer)
    accuracy = accuracy_score(y_test, y_pred)
    layer_accuracies3[layer_index] = accuracy

    with open(f'gradients/test_data4_{layer_index}.pkl', 'rb') as f:
        data_embeddings_test = pickle.load(f)
    with open(f'gradients/test_instruction4_{layer_index}.pkl', 'rb') as f:
        instruction_embeddings_test = pickle.load(f)
    x_test_layer = np.concatenate([data_embeddings_test, instruction_embeddings_test], axis=0)
    y_test = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))
    x_test_layer, y_test = shuffle(x_test_layer, y_test, random_state=42)
    y_pred = mlp.predict(x_test_layer)
    accuracy = accuracy_score(y_test, y_pred)
    layer_accuracies4[layer_index] = accuracy

for layer, accuracy in layer_accuracies1.items():
    print(f"Test Accuracy1 for Layer {layer}: {accuracy * 100:.2f}%")
for layer, accuracy in layer_accuracies2.items():
    print(f"Test Accuracy2 for Layer {layer}: {accuracy * 100:.2f}%")
for layer, accuracy in layer_accuracies3.items():
    print(f"Test Accuracy3 for Layer {layer}: {accuracy * 100:.2f}%")
for layer, accuracy in layer_accuracies4.items():
    print(f"Test Accuracy4 for Layer {layer}: {accuracy * 100:.2f}%")
