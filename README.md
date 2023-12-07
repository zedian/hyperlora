# Task-Agnostic Low-Rank Adapters for Unseen English Dialects

## Abstract

Large Language Models (LLMs) are trained on corpora disproportionally weighted in favor of Standard American English. As a result, speakers of other dialects experience significantly more failures when interacting with these technologies. In practice, these speakers often accommodate their speech to be better understood. Our work shares the belief that language technologies should be designed to accommodate the diversity in English dialects and not the other way around. However, prior works on dialect struggle with generalizing to evolving and emerging dialects in a scalable manner. To fill this gap, our method, HyperLoRA, leverages expert linguistic knowledge to enable resource-efficient adaptation via hypernetworks. By disentangling dialect-specific and cross-dialectal information, HyperLoRA improves generalization to unseen dialects in a task-agnostic fashion. Not only is HyperLoRA more scalable in the number of parameters, but it also achieves the best or most competitive performance across 5 dialects in a zero-shot setting. In this way, our approach facilitates access to language technology for billions of English dialect speakers who are traditionally underrepresented. 

## Installation

1. Create a virtual environment
```
conda create -n hyperlora python=3.7.13
conda activate hyperlora
```
2. Install packages
```
pip install -r requirements.txt
```
3. (For Training) Download multi-value dependencies
```
git clone https://github.com/SALT-NLP/multi-value.git
```
4. (For Training) Install spaCy English pipeline and nltk wordnet for Multi-VALUE
```
python -m spacy download en_core_web_sm
python 
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('cmudict')
>>> quit()
```
(Optional:) You may have to install neuralcoref and torch reparam from scratch
```
# Neuralcoref
git clone https://github.com/huggingface/neuralcoref.git/
cd neuralcoref;pip install -r requirements.txt;pip install -e .
# Pytorch Reparam
git clone https://github.com/ssnl/PyTorch-Reparam-Module.git
cd PyTorch-Reparam-Module/; python setup.py install
```

## Usage

<img width="784" alt="Training and Inference" src="https://github.com/zedian/hyperlora/assets/45089654/b1dca77a-3357-4df9-a834-7f4eaf0b4a23">

To train your own HyperLoRA, you can specify your training parameters and run
```
./run_hypernet_ot.sh
```

For evaluating HyperLoRA, you can specify testing data and run
```
./eval_hyperlora.sh
```

## Cite and Contact

```
@inproceedings{xiao2023taskagnostic,
      title={Task-Agnostic Low-Rank Adapters for Unseen English Dialects}, 
      author={Zedian Xiao and William Held and Yanchen Liu and Diyi Yang},
      booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      year={2023},
}
```

For any question please contact me at markxiao[at]stanford.edu! Thank you for your interest in our work!

## Acknowledgements

Code: The codebase was adapted from previous work on adapters for dialects at https://github.com/Helw150/tada
