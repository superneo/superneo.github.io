# Paper Summary: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

## Abstract
- BERT: Bidirectional Encoder Representations from Transformers
- trained using left/right contexts altogether, different from GPT1
- required only an additional output layer for various downstream tasks
- almost no need to modify the model architecture itself
- new SOTAs on 11 NLP tasks like GLUE/MultiNLI/SQuAD v1.1 QA test f1/SQuAD v2.0 test f1

## Introduction
- LM pre-training shown to be beneficial for sentence/token-level tasks(NLI, paraphrasing, NER, QA, etc.)
- 2 existing strategies of ELMo(feature-based) and GPT(fine-tuning) suffer from unidirectional learning
- BERT employs the MLM(masked language modeling) technique to fully utilize context in both directions
- and also use the NSP(next sentence prediction) task to jointly pre-train text-pair representations
- in short, bidirectional LM pre-training helps improve performance over unidirectional models like GPT1 and ELMO
- no need for task-specific architecture modification to achieve SOTA on many sentence/token-level tasks
- BERT set up new SOTA in 11 NLP tasks at the moment

## Related Work
- unsupervised feature-based approaches
  - learning word representations with non-neural/neural methods
  - pre-training word embeddings with methods like NNLM(left to right) or CBOW/Skip-gram(left/right context)
  - learning sentence/paragraph embeddings with next sentence ranking/next sentence generation/denoising autoencoder
  - ELMo and alikes to concatenate left-to-right and right-to-left contexts and learn context-sensitive representations
    - known to improve existing task-specific SOTAs in QA/NER/sentiment analysis but not that deeply bidirectional
- unsupervised fine-tuning approaches
  - pre-training from unlabeled text and fine-tuning for a supervised downstream task in modern researches
  - very few parameters need to be added and learned from scratch
  - left-to-right language modeling and auto-encoder objectives for pre-training GPT1 and alikes
- transfer learning from supervised data
  - if a large labeled data is ready then it's easy to pre-train a model on that data and transfer to downstream tasks

- Comparison of BERT, ELMo, and OpenAI GPT

<p align="center"><img src="../images/BERT/BERT_3.png" alt="Comparison of BERT, ELMo, and OpenAI GPT" width="1200"/></p>

## BERT
  - featured with 2 steps in its framework: pre-training and fine-tuning
  - unified architecture across different tasks: minimal difference between the pre-trained architecture and downstream

<p align="center"><img src="../images/BERT/BERT_1.png" alt="BERT - pre-training and fine-tuning" width="1200"/></p>

- Model Architecture
  - TBD
- Input/Output Representations
  - TBD

<p align="center"><img src="../images/BERT/BERT_2.png" alt="BERT - input representation" width="1200"/></p>

### pre-training BERT
- Task #1: Masked LM
  - TBD
- Task #2: Next Sentence Prediction (NSP)
  - TBD
- Pre-training data
  - TBD

### Fine-tuning BERT
- TBD

- Illustrations of Fine-tuning on Different Tasks

<p align="center"><img src="../images/BERT/BERT_4.png" alt="Illustrations of Fine-tuning on Different Tasks" width="1200"/></p>
