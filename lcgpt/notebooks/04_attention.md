---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: LightningcatcherGPT
  language: python
  name: lcgpt
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# 04 — GP Transform: Attention

Covers `karpathy.py` lines 120–136: seventeen lines in the middle of `gpt`, and the
only place in the entire model where information moves between positions.

Everything in notebook 03 acted on one position's vector in isolation — `linear`,
`rmsnorm`, the MLP, the embedding lookup, the output head. Run the model on a
single token and none of them behave differently. That is the test worth applying
here: attention is the one operation that fails it.

`softmax` was covered in notebook 01, lines 75–79. It reappears on line 132 doing a
different job — normalising over *positions* rather than over the vocabulary.

Code cells reproduce the source verbatim, dedented where a fragment sits inside a
function. Anything that is *not* from the source is marked as such.

+++

## 4.1 Queries, keys and values — lines 120–122

Three `linear` calls on the same input, with three different learned matrices.
Each produces a vector of width `n_embd` from the position's own state, and the
three are read as three different questions about it:

- **query** — what this position is looking for
- **key** — what this position offers as a match
- **value** — what this position contributes if matched

Nothing in the code enforces that reading. `wq`, `wk` and `wv` are initialised from
the same distribution and differ only in what the gradient does to them.

```{code-cell} ipython3
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
```

## 4.2 The cache — lines 123–124

The key and value for this position are appended to lists that persist across
positions. `keys[li]` therefore holds one entry per position seen so far, oldest
first.

This is the KV cache, and it is worth being clear about its status: it is not
architecture. It changes no output. Its purpose is that `gpt` is called once per
position, and without it every call would have to recompute the keys and values for
the entire prefix. `generate` allocates the lists fresh per sample; `train`
allocates them fresh per document.

```{code-cell} ipython3
keys[li].append(k)
values[li].append(v)
```

## 4.3 Splitting into heads — lines 125–130

The `n_embd`-wide vectors are sliced into `n_head` contiguous chunks of `head_dim`.
Each chunk attends independently, and the results are concatenated back to full
width by `x_attn.extend(head_out)`.

No parameters are added by this — the same `wq`, `wk`, `wv` produced the full-width
vectors, and heads are a reinterpretation of those coordinates. The point is that
one head computes exactly one distribution over positions, so a single head must
gather everything through one weighted average. Four heads give four, and they can
attend to different places at once.

```{code-cell} ipython3
x_attn = []
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

## 4.4 Attention scores — line 131

The dot product of this position's query with every cached key, one score per
position:

$$
s_t \;=\; \frac{q \cdot k_t}{\sqrt{d_h}}
$$

The $\sqrt{d_h}$ matters. A dot product of two $d_h$-dimensional vectors with
unit-scale entries grows like $d_h$, and feeding large numbers into a softmax
saturates it — one weight goes to 1, the rest to 0, and the gradient through it
vanishes. Dividing keeps the scores in a range where the softmax stays soft.

```{code-cell} ipython3
attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
```

## 4.5 Causal masking, by omission — line 131 again

A language model must not see the future. The usual implementation adds $-\infty$
to the scores above the diagonal before the softmax.

There is no such line here. `k_h` is built from `keys[li]`, and `keys[li]` holds
only positions $0 \ldots t$, because the cache is appended to one position at a
time and position $t{+}1$ has not run yet. The loop bound *is* the mask.

That equivalence is specific to this incremental formulation. Batched
implementations compute all positions at once and need the explicit $-\infty$; the
attention triangle in the demo below is the same constraint, made visible.

```{code-cell} ipython3
# Not from the source: an annotation on line 131.
#
#   attn_logits = [... for t in range(len(k_h))]
#                                    ^^^^^^^^^^
#                      k_h holds positions 0..t only, because the cache
#                      is appended to as generation proceeds — so this
#                      bound *is* the causal mask
```

## 4.6 Weights and the weighted sum — lines 132–134

The scores become a distribution over positions, and the head's output is the
average of the cached values under it:

$$
\alpha \;=\; \operatorname{softmax}(s)
\qquad
\mathrm{head} \;=\; \sum_t \alpha_t \, v_t
$$

This is the crossing point. Every other line in the model reads only `x`; this one
reads `v_h[t]` for every earlier `t`. The weights $\alpha$ are activations, not
parameters — they are recomputed for every position of every document and are never
stored, which is why seeing them requires the demo below.

Putting 4.1 through 4.6 together gives the whole operation:

$$
\operatorname{Attention}(q, K, V) \;=\; \operatorname{softmax}\!\left(\frac{q K^{\top}}{\sqrt{d_h}}\right) V
$$

```{code-cell} ipython3
attn_weights = softmax(attn_logits)
head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
x_attn.extend(head_out)
```

## 4.7 Output projection and residual — lines 135–136

The concatenated heads pass through a fourth learned matrix, `wo`, which is what
lets the model mix across head boundaries — without it each head's output would
occupy its own fixed slice of the channels forever.

Then the residual add from notebook 03 closes the sub-block, and the MLP begins.

```{code-cell} ipython3
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
x = [a + b for a, b in zip(x, x_residual)]
```

## 4.8 Watching one head attend

Not from the source. The attention weights are activations, so no trained model
contains them and `gpt` does not return them. To see them, the attention step has
to be run again with the weights kept.

The model is notebook 03's: 72 parameters, one layer, one head, trained on `aba`
three times and `abc` once. The cell below reproduces lines 111–131 exactly — the
same two `rmsnorm` calls, the same `wq`/`wk`, the same scaling — and prints the
softmax output at each position instead of discarding it.

Two things to look for. The result is **triangular**: row $t$ has exactly $t+1$
entries, which is §4.5's mask with nothing hidden. And every row **sums to 1**,
because a row is a softmax — attention redistributes a fixed budget rather than
choosing freely, so a position that attends more to one place necessarily attends
less to another.

```{code-cell} ipython3
# Not from the source: lines 111-131 re-run with the attention weights kept.
import sys; sys.path.insert(0, '..')   # karpathy.py lives one level up
import karpathy
from karpathy import rmsnorm, linear, softmax

docs = ['aba', 'aba', 'aba', 'abc']
config = karpathy.new_model_config(docs, verbose=False, seed=42,
                                   n_embd=2, n_head=1, block_size=4)
karpathy.train(config, docs, num_steps=3000, verbose=False)
names = config['uchars'] + ['BOS']

def attention_weights(config, prefix):
    """Re-run the layer-0 attention step, keeping the weights the model computes."""
    sd, head_dim = config['state_dict'], config['head_dim']
    ids = [config['BOS']] + [config['uchars'].index(c) for c in prefix]
    keys, rows = [], []
    for pos_id, token_id in enumerate(ids):
        x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id])]
        x = rmsnorm(x)                                    # line 114
        x = rmsnorm(x)                                    # line 119, inside the block
        q = linear(x, sd['layer0.attn_wq'])
        k = linear(x, sd['layer0.attn_wk'])
        keys.append(k)                                    # line 123
        scores = [sum(q[j] * kt[j] for j in range(head_dim)) / head_dim**0.5
                  for kt in keys]                         # line 131
        rows.append([p.data for p in softmax(scores)])    # line 132
    return ids, rows

ids, rows = attention_weights(config, 'aba')
labels = [names[i] for i in ids]

print("attention weights, layer 0 head 0")
print("row = the position doing the attending, column = the position attended to\n")
print("            " + "".join(f"{l:>8}" for l in labels))
for t, row in enumerate(rows):
    print(f"  pos {t} {labels[t]:>3}  " + "".join(f"{w:8.3f}" for w in row))
print("\nrow sums:", "  ".join(f"{sum(r):.3f}" for r in rows))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```
