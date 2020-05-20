# AttentionXML
[AttentionXML: Tree based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification](https://arxiv.org/abs/1811.01727)

## Requirements

See requirements.txt

## Datasets

|[EUR-Lex](https://drive.google.com/open?id=1EQaXWHYnihKv3ZEZ2pyM1dRJNn11IT_z)|[Wiki10-31K](https://drive.google.com/open?id=13ayVRMqfpzhMWKFDH1BvaQ2EFWB4UDQ-)|[AmazonCat-13K](https://drive.google.com/open?id=1CD_MATrUJC_ZgnIU4qghTSiT4WUZ_mnR)|[Amazon-670K](https://drive.google.com/open?id=1HraWMWfAfBP4PFVqDpy12dmL2kLHdoJh)|[Wiki-500K]()|[Amazon-3M](https://drive.google.com/open?id=1bhBcRO55oNk4LRRIexgpSOCQ7pEgTT5l)|
|---|---|---|---|---|---|

## Preprocess

Download the GloVe embedding (840B,300d) and convert it to gensim format(which can be loaded by gensim.models.KeyedVectors.load).

We also provide a converted GloVe embedding at [here](https://drive.google.com/file/d/10w_HuLklGc8GA_FtUSdnHT8Yo1mxYziP/view?usp=sharing). 

Run preprocess.py for train and test datasets with tokenized texts as follows:
```bash
PYTHONPATH=src python src/deepxml/preprocess.py \
--text-path data/EUR-Lex/train_texts.txt \
--label-path data/EUR-Lex/train_labels.txt \
--vocab-path data/EUR-Lex/vocab.npy \
--emb-path data/EUR-Lex/emb_init.npy \
--w2v-model data/glove.840B.300d.gensim

PYTHONPATH=src python src/deepxml/preprocess.py \
--text-path data/EUR-Lex/test_texts.txt \
--label-path data/EUR-Lex/test_labels.txt \
--vocab-path data/EUR-Lex/vocab.npy 
```

Or run preprocss.py including tokenizing the raw texts by NLTK as follows:
```bash
PYTHONPATH=src python src/deepxml/preprocess.py \
--text-path data/Wiki10-31K/train_raw_texts.txt \
--tokenized-path data/Wiki10-31K/train_texts.txt \
--label-path data/Wiki10-31K/train_labels.txt \
--vocab-path data/Wiki10-31K/vocab.npy \
--emb-path data/Wiki10-31K/emb_init.npy \
--w2v-model data/glove.840B.300d.gensim

PYTHONPATH=src python src/deepxml/preprocess.py \
--text-path data/Wiki10-31K/test_raw_texts.txt \
--tokenized-path data/Wiki10-31K/test_texts.txt \
--label-path data/Wiki10-31K/test_labels.txt \
--vocab-path data/Wiki10-31K/vocab.npy 
```


## Train and Predict

Train and predict AttentionXML without Tree:
```bash
PYTHONPATH=src python src/deepxml/main.py \
--data-cnf configure/datasets/EUR-Lex.yaml \
--model-cnf configure/models/AttentionXML-EUR-Lex.yaml 
```
Train and predict AttentionXML with Tree:
```bash
PYTHONPATH=src python src/deepxml/tree.py \
--data-cnf configure/datasets/Wiki-500K.yaml \
--model-cnf configure/models/FastAttentionXML-Wiki-500K.yaml
```

Or do prediction only with option "--mode eval".

## Ensemble

Train and predict with ann ensemble:
```bash
PYTHONPATH=src python src/deepxml/tree.py \
--data-cnf configure/datasets/Wiki-500K.yaml \
--model-cnf configure/models/FastAttentionXML-Wiki-500K.yaml -t 0
PYTHONPATH=src python src/deepxml/tree.py \
--data-cnf configure/datasets/Wiki-500K.yaml \
--model-cnf configure/models/FastAttentionXML-Wiki-500K.yaml -t 1
PYTHONPATH=src python src/deepxml/tree.py \
--data-cnf configure/datasets/Wiki-500K.yaml \
--model-cnf configure/models/FastAttentionXML-Wiki-500K.yaml -t 2
PYTHONPATH=src python src/deepxml/ensemble.py \
-p results/FastAttentionXML-Wiki-500K -t 3\
```

### Evaluation

```bash
PYTHONPATH=src python -W ignore src/deepxml/evaluation.py \
--results results/AttentionXML-EUR-Lex-labels.npy \
--targets data/EUR-Lex/test_labels.npy
```
Or get propensity scored metrics together:

```bash
PYTHONPATH=src python -W ignore src/deepxml/evaluation.py \
--results results/FastAttentionXML-Amazon-670K-labels.npy \
--targets data/Amazon-670K/test_labels.npy \
--train-labels data/Amazon-670K/train_labels.npy \
-a 0.6 \
-b 2.6

```

## Refrences
[AttentionXML: Tree based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification](https://arxiv.org/abs/1811.01727)
