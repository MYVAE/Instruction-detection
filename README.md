# Instruction-detection

## Contents
- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [Usage](#usage)
- [License](./LICENSE)


## Overview
The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. 
However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. 
We recognize that the success of IPI attacks fundamentally relies in the presence of instructions embedded within external content, which can alter the behavioral state of LLMs. Can effectively detecting such state changes help us defend against IPI attacks? 
In this paper, we propose a novel approach that takes external data as input and leverages the behavioral state of LLMs during both forward and backward propagation to detect potential IPI attacks. Specifically, we demonstrate that the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. 
By effectively combining these features, our approach achieves a detection accuracy of 99.60\% in the in-domain setting and 96.90\% in the out-of-domain setting, while reducing the attack success rate to just 0.12\% on the BIPIA benchmark.

## Repo Contents
- [src](./src): source code to reproduce results in the manuscript.
- [scripts](./scripts): scripts to run the experiments.
- [dataset](./dataset): save dataset.
- [hidden_states](./hidden_states): save hidden state features.
- [gradients](./gradients): save gradient features.

## Usage

### Download Dataset
The BIPIA dataset is available at https://github.com/microsoft/BIPIA, we use `benchmark/text_attack_train.json` and `benchmark/text_attack_test.json` in our evaluation.

The NewsArticle dataset is available at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GMFCTR

Please download the dataset and save at the `./dataset`

### Demo
#### Instruction Detection
```bash
# prepare gradient features
python src/prepare_gradient.py

# prepare hidden state features
python src/prepare_hidden_state.py

# perform instruction detection utilizing gradient features and hidden state features
python src/classification.py
```

#### Solely utilizing gradient features / hidden state features
```bash
# solely utilize gradient features to detect instruction (layer 13 is the best layer —— start from 0)
python src/test_gradient.py --start 13 --end 14

# solely utilize hidden state features to detect instruction (layer 14 is the best layer —— start from 1)
python src/test_hidden_state.py --start 14 --end 15
```
Parameters:
- start: First layer for evaluation.
- end: Last layer for evaluation.

#### Utilizing different layers of gradient features / hidden state features
```bash
# utilize gradient features of different layers to detect instruction
python src/test_gradient.py

# utilize hidden state features of different layers to detect instruction
python src/test_hidden_state.py
```

#### Utilizing different large language models
```bash
./run_llm.sh {model_path}
```
Parameters:
- model: The model for hidden state feature & gradient feature extraction. The model must be compatible with Hugging Face's transformers library and meet these specifications
    - support `output_hidden_states=True` parameter
    - implement gradient computation via `requires_grad=True`

#### Impact of paired response
```bash
./run_response.sh {paired_response}
```
Parameters:
- paired_response: The paired response used for the extraction of gradient features. It should be common responses to instructions, such as 'Sure', 'Hello', 'Yes' and 'I'm sorry'.

#### Impact of training dataset size
```bash
./run_data_size.sh {data_size}
```
Parameters:
- data_size: The dataset size used for training, should be an even number (half positive samples, half negative samples).

#### Impact of training dataset combination
```bash
./run_data_type.sh {data_type} {instruction_type}
```
Parameters:
- data_type: The external content dataset used for training. The options include: `["wiki", "news"]`.
- instruction_type: The instruction dataset used for training. The options include: `["1", "2"]`. `1` stands for LaMini-instruction dataset, `2` stands for BIPIA instruction dataset.