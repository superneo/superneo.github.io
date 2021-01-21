# paper summary: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

## Disclaimer
This is a simple coverage for the transformer paper.<br>
It'd be a brief and subjective summary of the original,<br>
with a bit of my own perspective,<br>
not a kind of line-by-line commentary or analysis.<br>

## model architecture

### encoder-decoder paring in brief
<img src="../images/transformer/fig_01.png" alt="original paper figure 1" width="400"/>

### transformer architecture in more details
<img src="../images/transformer/trx_arch_ext.png" alt="transformer architecture extension" width="450"/>

- in each encoder and decoder layer, the self-attention enables for each position to attend to all positions in principle
- it implements the seq-to-seq transduction masking all positions rightward of the current step in the decoder layers
- the topmost encoder output acts as the keys & values entering into each decoder layer

### scaled dot-product and multi-head attention
<img src="../images/transformer/fig_02.png" alt="original paper figure 2" height="400"/>

#### scaled dot-product attention
<img src="../images/transformer/scaled_dot_product_attn.png" alt="original paper formula" width="300"/>

- the commonly used attention method since [NMT by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf)
- it deals with the inner-product magnitude scale problem of large-sized vectors by introducing a scaling factor(1/sqrt(d_k))
- computationally cheaper than additive attention with appropriate application of a scaling factor as practical heuristic

#### multi-head attention
<img src="../images/transformer/multihead_attn.png" alt="original paper formula" width="400"/>

- it provides with multiple attention results using different parallel 'scaled dot-product attention' heads
- and it has a compute load almost the same as that of a single head attention with the full dimensionality
- a brilliant choice for both general performance regularity and computational efficiency

### position-wise feed-forward networks
<img src="../images/transformer/position_wise_ffn.png" alt="original paper formula" width="300"/>

- each layer in the encoder/decoder has a feed forward network with parameters shared across positions but unique per layer

### embeddings and softmax
- the same matrix is shared between the input/output embedding layers and the pre-softmax linear transformation
- and multiply those embedding layer weights by sqrt(d_model)

### positional encoding
<img src="../images/transformer/positional_encoding.png" alt="original paper formula" width="300"/>

- sinusoidal functions are used to inject the sequence order information to help the model learn to attend by relative positions
- any position with some fixed offset k, PE_(pos+k), can be represented as a linear function of PE_pos

## why self-attention
<img src="../images/transformer/tbl_01.png" alt="original paper formula" width="800"/>

- complexity per layer
  - self-attention layer has less computational complexity than recurrent layer if n is less than d which is most often the case
  - self-attention layer has much less complexity than conv layer (separable convolutions have complexity between self-attention only and self-attention + FFN)
  - when n is very large, restricted self-attention could be an alternative but it increases the max path length to O(n/r)
- sequential operations
  - self-attention layer has a constant number of sequential operations as conv layer or restricted self-attention layer do while recurrent layer does the worst
- maximum path length
  - a single self-attention layer simply connects all positions to each other which no other layer type can do
