# paper summary: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

## Disclaimer
This is a simple coverage for the transformer paper.<br>
It'd be a brief and subjective summary of the original,<br>
with a bit of my own perspective,<br>
not a kind of line-by-line commentary or analysis.<br>

## model architecture

### encoder-decoder paring in brief
<img src="../images/transformer/fig_01.png" alt="original paper figure 1" width="300"/>

### transformer architecture in more details
<img src="../images/transformer/trx_arch_ext.png" alt="transformer architecture extension" width="400"/>

### scaled dot-product and multi-head
<img src="../images/transformer/fig_02.png" alt="original paper figure 2" height="300"/>

#### scaled dot-product attention
- the commonly used attention method since [NMT by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf)
- it deals with the inner-product magnitude scale problem of large-sized vectors by introducing a scaling factor(1/sqrt(d_k))
- computationally cheaper than additive attention with appropriate application of a scaling factor as practical heuristic

#### multi-head attention
- it provides with multiple attention results using different parallel 'scaled dot-product attention' heads
- and it has a compute load almost the same as that of a single head attention with the full dimensionality
- a brilliant choice for both general performance regularity and computational efficiency

