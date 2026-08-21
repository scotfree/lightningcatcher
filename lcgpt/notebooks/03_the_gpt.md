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

# 03 — The GPT

Covers `karpathy.py` lines 70–73, 81–105 and 107–146 minus the attention interior:
the model that has been standing in as `logit_model` for two notebooks.

`gpt` is a function from `(config, token_id, pos_id, keys, values)` to a logit per
vocabulary entry — the same signature the counting model and the bigram table
satisfied. What changes is the middle. This notebook walks the outside of that
middle: what goes in, what comes out, and the two per-position transforms that
wrap the attention step. Notebook 04 opens the step itself, lines 120–136.

The structural claim worth holding on to: **every operation here acts on one
position's vector alone**, with the same weights at every position. Run the model
on a single token and nothing in this notebook behaves differently. Only attention
knows there is a sequence.

Code cells reproduce the source verbatim, dedented where a fragment sits inside a
function. Anything that is *not* from the source is marked as such.

+++

## 3.1 `linear` — lines 81–82

A matrix-vector product, written as a list comprehension. `w` is a list of output
rows, each the same length as `x`, so the result has one entry per row:

$$
y_o \;=\; \sum_i w_{oi}\, x_i
$$

No bias term — GPT-2 has one, this does not. Every learned transform in the model
is a call to this function.

```{code-cell} ipython3
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

## 3.2 `rmsnorm` — lines 70–73

Rescales a vector to roughly unit scale, dividing by the root mean square of its
own entries:

$$
\operatorname{rmsnorm}(x)_i \;=\; \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \varepsilon}}
$$

GPT-2 uses layernorm, which also subtracts the mean and applies a learned gain.
This drops both, so the function has no parameters at all.

The $\varepsilon$ keeps the reciprocal finite on an all-zero vector. Everything
here is per-position: the sum runs over the *channels* of one token's vector, never
across tokens.

```{code-cell} ipython3
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

## 3.3 Initialising the parameters — lines 84–105

Every weight in the model, created here and never created anywhere else. Gaussian
noise at `std=0.08` — small enough that the initial logits are nearly uniform,
which is why training starts near $\log V$.

Three matrices exist per model:

- **`wte`** — one row per vocabulary entry, the token's learned vector.
- **`wpe`** — one row per position, the position's learned vector.
- **`lm_head`** — one row per vocabulary entry, turning a vector back into logits.

and six more per layer: four for attention (`wq`, `wk`, `wv`, `wo`) and two for the
MLP, which widens to `4 * n_embd` and back. That `4` is inherited from GPT-2 and is
where most of the parameters live.

`head_dim` is set here because it is derived, not chosen: the width is split evenly
across heads, so `n_embd` must divide by `n_head`.

```{code-cell} ipython3
def init_params(config, seed=None):
    """Initialise a fresh set of weights in `config`, in place."""
    random.seed(seed)
    config['head_dim'] = config['n_embd'] // config['n_head']    # derived dimension of each head

    # Initialize the parameters, to store the knowledge of the model
    n_embd, vocab_size, block_size = config['n_embd'], config['vocab_size'], config['block_size']

    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(config['n_layer']):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    config['state_dict'] = state_dict
    return config


```

## 3.4 Entering the model: embeddings — lines 107–114

A token id is an integer; the model needs a vector. `wte[token_id]` is that vector,
looked up rather than computed — a row of a matrix that gradient descent is free to
move.

Position enters the same way, and the two are simply **added**:

$$
x \;=\; \mathrm{wte}[\,t\,] \;+\; \mathrm{wpe}[\,p\,]
$$

Addition rather than concatenation means "which token" and "where" share the same
$d$ channels and the model has to learn to disentangle them. Without `wpe` the
model would be permutation-invariant: attention has no inherent order, so position
must be supplied as data.

The `rmsnorm` on line 114 is not redundant with the one inside the block. The
comment says why — the residual connection carries `x` forward unnormalised, so
this one is the only thing bounding the scale entering the stack.

```{code-cell} ipython3
def gpt(config, token_id, pos_id, keys, values):
    state_dict = config['state_dict']
    n_layer, n_head, head_dim = config['n_layer'], config['n_head'], config['head_dim']

    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection
```

## 3.5 The layer loop and the residual stream — lines 116–119

`x_residual = x` before each sub-block, and `x = [a + b for a, b in zip(x, x_residual)]`
after it. The pattern appears twice per layer, around attention and around the MLP:

$$
x \;\leftarrow\; x \;+\; \mathrm{sublayer}(\mathrm{rmsnorm}(x))
$$

Two consequences. The sub-block learns a *correction* to `x` rather than a
replacement, so a freshly initialised layer that outputs near-zero is close to the
identity and does no harm. And in the backward pass the `+` sends gradient down
both branches unchanged, so the path from the loss to the embeddings stays short no
matter how many layers are stacked.

Normalising *inside* the branch and adding the un-normalised `x` back is the
pre-norm arrangement.

```{code-cell} ipython3
for li in range(n_layer):
    # 1) Multi-head Attention block
    x_residual = x
    x = rmsnorm(x)
```

## 3.6 The MLP block — lines 137–143

Attention has finished; this is the other half. Widen to `4 * n_embd`, apply ReLU,
project back:

$$
\mathrm{MLP}(x) \;=\; W_2 \,\max(0,\, W_1 x)
$$

GPT-2 uses GeLU; this uses ReLU, which `Value` already has. It is the only
nonlinearity in the model — everything else is linear or a normalisation, and a
stack of linear maps would collapse into one.

This block is where the model does its per-position work: attention gathers
information from other positions, and the MLP is what processes what was gathered.
It holds roughly two thirds of the parameters.

```{code-cell} ipython3
# 2) MLP block
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]
```

## 3.7 The output head — lines 145–146

One more `linear`, from `n_embd` back up to `vocab_size`, and the function returns.
Nothing normalises these — softmax happens in the caller, whether that is `train`
computing a loss or `generate` drawing a sample.

`lm_head` and `wte` have identical shapes. GPT-2 ties them to the same matrix;
this keeps them separate, which costs `vocab_size * n_embd` parameters and is one
of the differences the header comment refers to.

```{code-cell} ipython3
logits = linear(x, state_dict['lm_head'])
return logits
```

## 3.8 A model small enough to read

Not from the source. Every section above described a matrix and asked you to accept
that these matrices *are* the model's knowledge. At the default size that is an
assertion — 4,192 numbers is not something you can look at. So shrink the model
until it is.

`n_embd=2, n_head=1, block_size=4` over a three-letter alphabet gives **72
parameters**, every one of which fits on screen. `seed=` fixes the initialisation
so the numbers stay put when the cell is re-run.

The corpus is four documents, deliberately lopsided: `aba` three times and `abc`
once. So after the prefix `ab`, the corpus says `a` three times in four and `c`
once. If the weights really do encode the corpus, the model's predicted
distribution should be exactly that:

$$
P_\theta(t \mid \texttt{ab}) \;=\; \hat{P}_{\text{corpus}}(t \mid \texttt{ab})
\;=\; \left(\tfrac{3}{4},\, 0,\, \tfrac{1}{4},\, 0\right)
$$

Nothing forces this. It is what training is *for*.

```{code-cell} ipython3
# Not from the source: the whole model, small enough to print.
import sys; sys.path.insert(0, '..')   # karpathy.py lives one level up
import karpathy

docs = ['aba', 'aba', 'aba', 'abc']    # after "ab": 'a' 3 times in 4, 'c' once
config = karpathy.new_model_config(docs, verbose=False, seed=42,
                                   n_embd=2, n_head=1, block_size=4)
names = config['uchars'] + ['BOS']

def next_token_probs(config, prefix):
    """Run the model over `prefix` and return its distribution over what comes next."""
    keys, values = [[] for _ in range(config['n_layer'])], [[] for _ in range(config['n_layer'])]
    ids = [config['BOS']] + [config['uchars'].index(c) for c in prefix]
    for pos_id, token_id in enumerate(ids):
        logits = karpathy.gpt(config, token_id, pos_id, keys, values)
    return [p.data for p in karpathy.softmax(logits)]

show = lambda d: "  ".join(f"{n}={v:.2f}" for n, v in zip(names, d))

total = sum(len(row) for mat in config['state_dict'].values() for row in mat)
print(f"{total} parameters, vocabulary {config['vocab_size']}\n")
print("before training, after 'ab':", show(next_token_probs(config, 'ab')))

karpathy.train(config, docs, num_steps=3000, verbose=False)

print("after  training, after 'ab':", show(next_token_probs(config, 'ab')))
print("the corpus itself          :", show([0.75, 0.0, 0.25, 0.0]))

print("\nevery weight in the model:")
for name, mat in config['state_dict'].items():
    print(f"\n{name}")
    for row in mat:
        print("   ", "  ".join(f"{p.data:+.4f}" for p in row))
```
