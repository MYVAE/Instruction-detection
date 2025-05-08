import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch.nn.functional as F
from utils import get_instruction_train, get_data_train, insert_middle, get_instruction_test1, get_instruction_test2, get_data_wiki, get_data_news


model_name = "meta-llama/Llama-3.1-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)


def get_input_gradient(input_text, target_text="Yes"):
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


instructions_train = get_instruction_train()
data_train = get_data_train()
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

instructions_test = get_instruction_test1()
data_test = get_data_wiki()
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

data_test = get_data_news()
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

instructions_test = get_instruction_test2()
data_test = get_data_wiki()
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


data_test = get_data_news()
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