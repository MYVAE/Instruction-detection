import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.utils import shuffle
from tqdm import tqdm
from utils import get_instruction_train, get_data_train, insert_middle, get_instruction_test1, get_instruction_test2, get_data_wiki, get_data_news


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


instructions_train = get_instruction_train()
data_train = get_data_train()
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


instructions_test = get_instruction_test1()
data_test = get_data_wiki()
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


data_test = get_data_news()
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

instructions_test = get_instruction_test2()
data_test = get_data_wiki()
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

data_test = get_data_news()
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