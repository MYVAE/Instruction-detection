import json
import pandas as pd
import random
from nltk.tokenize.punkt import PunktSentenceTokenizer
from datasets import load_dataset


def insert_middle(context: str, attack: str, random_state: int = None):
    rng = random.Random(random_state)
    sentence_indexes = list(PunktSentenceTokenizer().span_tokenize(context))
    start, _ = rng.sample(sentence_indexes, k=1)[0]

    return " ".join([context[:start], attack, context[start:]])

def process(text):
    last_period = text.rfind('.')
    if last_period != -1:
        return text[:last_period + 1].strip()
    else:
        return text.strip()

def get_instruction_1_train():
    ds = load_dataset("MBZUAI/LaMini-instruction")
    data = ds["train"]["instruction"]
    return data[:100]

def get_instruction_2_train():
    instructions = []
    with open("dataset/text_attack_train.json", 'r') as f:
        json_data = json.load(f)
        for key in json_data:
            instructions.extend(json_data[key])
    return instructions

def get_instruction_1_test():
    ds = load_dataset("MBZUAI/LaMini-instruction")
    data = ds["train"]["instruction"]
    return data[500:1500]

def get_instruction_2_test():
    instructions = []
    with open("dataset/text_attack_test.json", 'r') as f:
        json_data = json.load(f)
        for key in json_data:
            instructions.extend(json_data[key])
    return instructions

def get_data_wiki_train():
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    data = ds["train"]["text"]
    data = [d.replace("\n", " ") for d in data]
    data = [process(d) for d in data if len(d) <= 5000]
    return data[:100]

def get_data_news_train():
    df = pd.read_csv('dataset/NewsArticles.csv', encoding='ISO-8859-1')
    data = df['text'].tolist()
    data = [item for item in data if isinstance(item, str)]
    data = [d.replace("\n", " ") for d in data]
    data = [d for d in data if len(d) <= 5000]
    return data[:100]

def get_data_wiki_test():
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    data = ds["train"]["text"]
    data = [d.replace("\n", " ") for d in data]
    data = [process(d) for d in data if len(d) <= 5000]
    return data[500:1500]

def get_data_news_test():
    df = pd.read_csv('dataset/NewsArticles.csv', encoding='ISO-8859-1')
    data = df['text'].tolist()
    data = [item for item in data if isinstance(item, str)]
    data = [d.replace("\n", " ") for d in data]
    data = [d for d in data if len(d) <= 5000]
    return data[500:1500]