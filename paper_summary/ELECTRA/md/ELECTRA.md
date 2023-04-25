# Paper Summary: [ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS](https://arxiv.org/pdf/2003.10555.pdf)
  - [github repo(google-research/electra)](https://github.com/google-research/electra)
  - [Google Research blog post(More Efficient NLP Model Pre-training with ELECTRA)](https://ai.googleblog.com/2020/03/more-efficient-nlp-model-pre-training.html)

## Abstract
  - BERT MLM produces good results for downstream NLP tasks but quite expensive
  - replaced token detection(RTD) as an alternative for better sample/compute-efficiency
  - RTD replaces some input tokens with plausible alternatives sampled from a small generator
  - then train a discriminator to predict whether each input token was replaced or original
  - RTD, using all input tokens, is proven experimentally to be more efficient than MLM
  - RTD learns better contextual representations than BERT given the same model size/data/compute
  - a small ELECTRA model trained on 1 GPU for 4 days defeats GPT(using 30x more compute) on the GLUE benchmark
  - comparable performance with < 1/4 of compute for RoBERTa/XLNet and better with the same amount of compute

## Introduction
  - MLM exceled predecessors using bidirectional context but incurred huge compute cost(ex. 15% input masking)
  - RTD is an alternative pre-training task to train a model to distinguish real input tokens from replaced ones
    - it corrupts the input by replacing some tokens with samples from a proposal distribution like small MLM output
    - it doesn't suffer from the pre-training/fine-tuning discrepancy for the [MASK] token just as BERT does
    - then pre-train the network as a discriminator that predicts for every token if it's original or replaced
    - it's more computationally efficient than MLM to learn from all input tokens, not just a small subset
    - and the generator corrputing tokens is trained with maximum likelihood, not adversarially as for GAN
  - they call the approach ELECTRA: Efficiently Learning an Encoder that Classifies Token Replacements Accurately
    - it's shown to train much faster than BERT and achieve higher accuracy on downstream tasks when fully trained
  - they train ELECTRA models of various sizes and evaluate their downstream performance vs their compute requirement
    - experiments on the GLUE benchmark and SQuAD question answering benchmark
    - ELECTRA outperforms MLM-based BERT/XLNet given the same model size/data/compute(Figure 1)
    - ELECTRA-Small, trained on 1 GPU in 4 days, beats a small BERT by 5 GLUE points and a much larger GPT model
    - ELECTRA-Large, with fewer params and 1/4 of training compute, matches RoBERTa/XLNet comparably
    - training ELECTRA-Large further even defeats ALBERT on GLUE and sets a new SOTA for SQuAD 2.0
  - in total, RTD is more compute/param-efficient than MLM-based methods for language representation learning

<p align="center"><img src="../images/ELECTRA_fig_01.png" alt="RTD vs MLM" width="800"/></p>

## Method

<p align="center"><img src="../images/ELECTRA_fig_02.png" alt="ELECTRA architecture" width="800"/></p>

  - ELECTRA approach trains 2 transformer encoder stacks: G(generator) and D(discriminator)
  - both map an input token seq **x** = [x<sub>1</sub>,...,x<sub>n</sub>] into a context vector reps seq h(**x**) = [h<sub>1</sub>,...,h<sub>n</sub>]
  - for x<sub>t</sub> = [MASK], G outputs a probability for generating a particular token x<sub>t</sub> with a softmax layer:

<p align="center"><img src="../images/ELECTRA_exp_01.png" alt="generator softmax probability" width="600"/></p>

  - 'e' denotes token embeddings
  - for a position 't', D predicts if the token x<sub>t</sub> is original or replaced with a sigmoid output layer:

<p align="center"><img src="../images/ELECTRA_exp_02.png" alt="discriminator sigmoid" width="300"/></p>

  - G is trained to perform masked language modeling (MLM)
  - D is trained to distinguish tokens in the data from tokens replaced by generator samples
  - model inputs are constructed according to:

<p align="center"><img src="../images/ELECTRA_exp_03.png" alt="model input construction" width="700"/></p>

  - and the loss functions:

<p align="center"><img src="../images/ELECTRA_exp_04.png" alt="loss functions" width="800"/></p>

  - although similar to the GAN training objective, there are several key differences:
    - if G generates the correct token, that token is considered 'real' instead of 'fake'
    - G is trained with maximum likelihood rather than being trained adversarially to fool D
      - applying GAN to text is hard as it's impossible to back-propagate through sampling from G
      - detouring this with reinforcement learning to train G performed worse than maximum-likelihood training
    - G doesn't get a noise vector as input which is typical for a GAN
  - so we minimize the combined loss over a large corpus X of raw text:
    - approximate the expectations in the losses with a single sample
    - don’t back-propagate the D loss through G(impossible due to the G sampling step)

<p align="center"><img src="../images/ELECTRA_exp_05.png" alt="combined loss" width="350"/></p>

  - after pre-training, we throw out G and fine-tune D on downstream tasks

## Experiments
### Experimental Setup
  - evaluate on the GLUE benchmark and SQuAD dataset
    - GLUE: RTE, MNLI, QNLU, MRPC, QQP, STS, SST, CoLA
    - SQuAD: text span selection for question answering(SQuAD 1.1) + unanswerability beyond 1.1(SQuAD 2.0)
    - evaluation metrics: Spearman correlation/Mathews correlation/accuracy for GLUE, Exact Match/F1 for SQuAD
  - pre-train on the same data as BERT for most experiments consisting of 3.3B tokens from Wikipedia + BooksCorpus
    - for ELECTRA-Large pre-train on XLNet data extending the BERT dataset to 33B tokens(ClueWeb/CommonCrawl/Gigaword)
  - the same model architecture and most hyper-params as BERT's
    - simple linear classifiers on top of ELECTRA for fine-tuning on GLUE
    - question-answering module from XLNet on top of ELECTRA from SQuAD 1.1 & 2.0('answerability' classifier added)
  - some evaluation datasets are small enough to incur substantial variance by the random seed
    - report the median of 10 fine-tuning runs from the same pre-trained checkpoint for each result
    - results are on the dev set by default
### Model Extensions
#### Weight Sharing
  - TBD

<p align="center"><img src="../images/ELECTRA_fig_03.png" alt="generator size and training methods" width="800"/></p>

#### Smaller Generators
  - TBD
#### Training Algorithms
  - TBD
### Small Models
  - TBD
### Large Models
  - TBD
### Efficiency Analysis
  - TBD
#### ELECTRA 15%
  - TBD
#### Replace MLM
  - TBD
#### All-Tokens MLM
  - TBD

<p align="center"><img src="../images/ELECTRA_tbl_05.png" alt="compute-efficiency experiments" width="800"/></p>

<p align="center"><img src="../images/ELECTRA_fig_04.png" alt="ELECTRA vs BERT for different sizes" width="800"/></p>

## Related Work
### Self-Supervised Pre-training for NLP
  - for word representations
    - Word2Vec, GloVe, etc.
  - and language modeling
    - BERT/MASS/UniLM/ERNIE/SpanBERT/XLNet/TinyBERT/MobileBERT/etc.
### Generative Adversarial Networks
  - effective at generating high-quality synthetic data
  - been applied to text data, though falling behind standard MLE
  - MaskGAN seemingly alike with a generator to fill in the deleted input tokens
### Contrastive Learning
  - learn to distinguish real from negative(fake) samples in multi-modalities
  - ELECTRA is particularly related to Noise-Contrastive Estimation(NCE)
    - can be viewed as a massively scaled-up Continous Bag-of-Words(CBOW)
    - reminiscent to predict an input token given surrounding context
    - and negative sampling renders the learning task as a binary cls(real vs fake)
    - but differs with  bag-of-vectors from unigram dist. vs transformer encoder

## Conclusion
  - RTD as a self-supervised task for language representation learning
  - to train a text encoder to distinguish original input tokens from negative samples
  - RTD is more data/downstream-task efficient than MLM
  - the writers hope ELECTRA will make pre-training large models more accessible
  - NLP pre-training needs to focus on compute-efficiency as well as absolute performance

--------------------------------------------------------------------------------------------

## Pre-Training Details
  - the following details apply both to ELECTRA and BERT baselines
    - hyper-params mostly the same as BERT
    - λ(the discriminator objective weight in the loss) set to 50
    - dynamic token masking(different masking pattern for every input seq)
    - no NSP(who the heck would use it?)
    - 25% of masking(not 15%) for ELECTRA-Large to lower the generator accuracy
    - the best learning rate for the Base/Small models picked out of [1e-4, 2e-4, 3e-4, 5e-4]
    - and λ out of [1, 10, 20, 50, 100] in early experiments

## Fine-Tuning Details
  - TBD
