# Instruction-detection

## Contents
- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [Installation Guide](#installation-guide)
- [License](./LICENSE)


## Overview
The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. 
However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. 
We recognize that the success of IPI attacks fundamentally relies in the presence of instructions embedded within external content, which can alter the behavioral state of LLMs. Can effectively detecting such state changes help us defend against IPI attacks? 
In this paper, we propose a novel approach that takes external data as input and leverages the behavioral state of LLMs during both forward and backward propagation to detect potential IPI attacks. Specifically, we demonstrate that the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. 
By effectively combining these features, our approach achieves a detection accuracy of 99.60\% in the in-domain setting and 96.90\% in the out-of-domain setting, while reducing the attack success rate to just 0.12\% on the BIPIA benchmark.

## Repo Contents
- [utils.py](./src/utils.py): necessary functions
- [prepare_gradient.py](./src/prepare_gradient.py): prepare gradient features
- [prepare_hidden_state.py](./src/prepare_hidden_state.py): prepare hidden state features
- [classification.py](./src/classification.py): conduct instruction detection

## Installation Guide

### Download Dataset
The BIPIA dataset is available at https://github.com/microsoft/BIPIA, we use benchmark/text_attack_test.json in our evaluation.

The NewsArticle dataset is available at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GMFCTR

Please download the dataset and save at the ./dataset

### Run Experiments
```bash
python src/prepare_gradient.py
python src/prepare_hidden_state.py
python src/classification.py
```