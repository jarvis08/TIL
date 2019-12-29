# Skip-grams

Variants of Skip-grams.

<br>

## Hierarchical Softmax

$p(w|w_{I}) = \prod_{j=1}^{L(w) - 1} \sigma ([\![ n(w, j+1) = ch(n(w, j))]\!]) \cdot v\prime _{n(w, j)}^{T} v_{w_{I}}$

<br>

<br>

## Negative Sampling

$log σ(v\prime_{w_{O}} ^{⊤} v_{w_{I}}) + ∑\mathbb{E}_{w_{i}~P_{n}(w)}[log σ(−v\prime _{w_{i}}^{⊤} v w_{I})]$

<br>

<br>

## Subsampling of Frequent Words

$P(w_{i}) = 1 - \sqrt{\frac{t}{f(w_{i})}}$

