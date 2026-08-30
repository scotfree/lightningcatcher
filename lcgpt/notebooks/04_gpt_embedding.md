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

# 04 — GP Transform: Embedding

The outer part of the Transform is simple embedding. 
We map a token at a position into a fancy vector space.
Then we do a bunch of fancy work inside hub space to get a new embedded vector.
Then we map it back to a token (which we'll use as the token for the /next/ position in our block sequence.

We'll talk about the fancy work that happens in the hub space in the next Notebook - here we're just getting to and from that space.

**Notes**
* `gpt` is a function from `(config, token_id, pos_id, keys, values)` to a logit per
vocabulary entry — the same signature the counting model and the bigram table
satisfied. What changes is the middle. This notebook walks the outside of that
middle: what goes in, what comes out, and the two per-position transforms that
wrap the attention step. Notebook 05 opens the step itself, lines 120–136.

* The structural claim worth holding on to: **every operation here acts on one
position's vector alone**, with the same weights at every position. Run the model
on a single token and nothing in this notebook behaves differently. Only attention
knows there is a sequence.

* Covers `karpathy.py` lines 70–73, 81–105 and 107–146 minus the attention interior:
the model that has been standing in as `logit_model` for two notebooks.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## 4.1 `linear` — lines 81–82

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

## 4.2 `rmsnorm` — lines 70–73

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

## 4.4 Entering the model: Embeddings — lines 107–114

We start by mapping a token and a position (relative to a block), both integers, into a vector in our "hub space", a vector space of dimension `n_embd`. These two mappings are stored as a `n_embd` row per id.



Then these two hub vectors are simply **added** to yeild $x$, the embedded Vector for this (token, position) pair:

$$
x \;=\; \mathrm{wte}[\,t\,] \;+\; \mathrm{wpe}[\,p\,]
$$
and then Root Mean Square normalize it.

Notes:

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

Now that we have `x` we'll apply the "Forward Pass" to modify it (in place) in the hub space.
When we're done with that, we'll simply need to map back to the token space:

## 4.7 The output head — lines 145–146

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

## 4.8 A model small enough to read

Not from the source. Every section above described a matrix and asked you to accept
that these matrices *are* the model's knowledge. At the default size that is an
assertion — 4,192 numbers is not something you can look at. So shrink the model
until it is.

TODO: just read a simple, likely untrained model here. We just want to show the embedding...

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

## 4.3 Initialising the Model — lines 84–105

Every weight in the model, created here and updated in `train()`. This is also where we start to see the shape and flow of the model itself, so we'll talk a bit about what these matrices are. We start with three model level matrices providing the Embedding itself. 

### Embedding
We embed vectors in two spaces - `token` and `position-in-block` - into a common `hub space` of dimension `n_embd`, then back to tokens.

#### Map Token Embedding ("mte")
Map a token_id in our vacabulary (<= vacab_size) to a vector in our "Hub Space" - weights across `n_embd` logits. One row for each token_id, each row has `n_embd` weights, so a `vocab_size x n_embd` matrix.

#### Map Position Embedding ("mpe")
Map a position in a block (<= block_size) to a vector in our "Hub Space" - weights across `n_embd` logits. One row for each position_id, each row has `n_embd` weights, so a `block_length x n_embd` matrix.

#### Language Model Head ("lm_head")
Map an embedded_vecotr in our `hub space` back to a vector of logits in vocabulary space. `n_embd x vocab_size`

### Attention

and six more per layer: four for attention (`wq`, `wk`, `wv`, `wo`) and 

### Multiple Layer Perceptron ("MLP")
MLP which widens to `4 * n_embd` and back. That `4` is inherited from GPT-2 and is
where most of the parameters live.

`head_dim` is set here because it is derived, not chosen: the width is split evenly
across heads, so `n_embd` must divide by `n_head`.

We initialize `a priori` with Gaussian noise at `std=0.08` — small enough that the initial logits are nearly uniform,
which is why training starts near $\log V$.

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

## 4.9 The embedding on its own

Not from the source. §4.8 shrank the model until its weights fit on screen. This one
removes the model and keeps only the embedding: no attention, no MLP, `n_layer=0`,
and no training at all — every number is set by hand.

`logit_model=` is what makes that possible. `train` and `generate` accept any
function with the signature `(config, token_id, pos_id, keys, values) -> logits`, so
`gpt` can be replaced by the two steps that matter here:

$$
x \;=\; \mathrm{mte}[t] \,+\, \mathrm{mpe}[p]
\qquad\qquad
\text{logits} \;=\; \mathrm{lm\_head}\;x \;+\; \beta
$$

Look up, add, project back. `keys` and `values` are accepted and ignored — there is
no attention to cache for. `mpe` is zeros throughout, so position plays no part; the
whole model is "which token am I", nothing else.

### Four tokens, one dimension

Put `A`, `B`, `C`, `D` at $1, 2, 3, 4$ on a single axis. Four distinct points on a
line, and the read-out sends each one to the next: the model generates `ABCDABCD`.

The read-out is worth looking at closely. Each token's logit is a straight line in
$x$:

$$
\text{logits}_i(x) \;=\; m_i\,x + \beta_i
$$

so the prediction — the argmax — is the **upper envelope** of five straight lines.
An upper envelope of $V$ lines can have up to $V$ segments, and here the winner
changes at $x = 1.5,\ 2.5,\ 3.5$: `B` owns the stretch containing `A`'s point, `C`
owns `B`'s, and so on around the cycle. `BOS` shares `D`'s point so the first step
predicts `A`, and its line sits below everything so it can never win — otherwise
sampling would stop at once, which is what line 269 does when `BOS` comes up.

### What the bias is doing

Drop $\beta$ and the same four-cycle becomes impossible in one dimension — which is
worth stating carefully, because the embedding is not where it fails.

Placing four tokens at distinct points on a line is fine. The map
$x \mapsto \mathrm{lm\_head}\,x$ into logit space is injective too, so no information
is lost. What fails is the **argmax on top of it**. Without a bias the reachable
logit vectors are $\{x \cdot \mathrm{lm\_head} : x \in \mathbb{R}\}$ — a line through
the *origin*. Multiplying by a positive $x$ scales every coordinate by the same
factor and cannot change which is largest; a negative $x$ reverses the order, once.
So over all $x$, and for any weights, the argmax takes exactly **two** values: the
largest entry of `lm_head` and the smallest.

Adding $\beta$ lifts that line off the origin, and an affine line through logit space
can cross as many argmax regions as there are tokens. That is the whole difference.

`karpathy.py` has no bias — `linear` on line 81 is a bare dot product, and GPT-2's
output head has no bias either. So for the model in this notebook the constraint is
real, and the way out is a second channel rather than an intercept.

### Bias-free, in two dimensions

With two channels the tokens fit on the unit circle, 90° apart. Give each token's
`lm_head` row its *predecessor's* vector; then

$$
\langle \mathrm{lm\_head}_i,\; x_t \rangle \;=\; \cos\!\big(\theta_{i-1} - \theta_t\big)
$$

which is largest exactly when token $i$ follows token $t$. The whole model is a 90°
rotation, it uses no bias, and it generates the same `ABCDABCD`.

The general statement behind both halves: $x \mapsto \mathrm{lm\_head}\,x$ has rank at
most `n_embd`, so the reachable logits lie in an `n_embd`-dimensional subspace of the
`vocab_size`-dimensional logit space. This is the **softmax bottleneck**, and it is
one concrete thing `n_embd` buys — not "capacity" vaguely, but directions the output
head can tell apart.

`temperature=0.01` makes sampling effectively an argmax, so the printed strings show
what the weights say rather than what the sampler happened to draw.

```{code-cell} ipython3
# Not from the source: the embedding machinery on its own, every weight set by hand.
import sys; sys.path.insert(0, '..')   # karpathy.py lives one level up
import karpathy
from karpathy import Value

def embedding_only(vocab, mte, lm_head, bias=None, block_size=8):
    """A config carrying nothing but the embedding matrices. No layers, no training."""
    n_embd = len(mte[0])
    config = dict(n_layer=0, n_embd=n_embd, n_head=1, head_dim=n_embd,
                  block_size=block_size, token_type='letter',
                  uchars=list(vocab), vocab_size=len(vocab) + 1, BOS=len(vocab))
    config['state_dict'] = {
        'wte':     [[Value(v) for v in row] for row in mte],
        'wpe':     [[Value(0.0) for _ in range(n_embd)] for _ in range(block_size)],  # unused
        'lm_head': [[Value(v) for v in row] for row in lm_head],
    }
    # karpathy.py's `linear` has no bias term; this is the demo's own addition.
    config['lm_head_bias'] = [Value(b) for b in (bias or [0.0] * (len(vocab) + 1))]
    return config

def embed_only(config, token_id, pos_id, keys, values):
    """A `logit_model` with no transformer in it: look up, add position, read out."""
    sd = config['state_dict']
    x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id])]     # lines 111-113
    return [sum(w * xi for w, xi in zip(row, x)) + b                       # line 145, plus bias
            for row, b in zip(sd['lm_head'], config['lm_head_bias'])]

def show(config, title):
    names = config['uchars'] + ['BOS']
    fmt = lambda row: "(" + ", ".join(f"{p.data:+g}" for p in row) + ")"
    print(f"{title}    n_embd={config['n_embd']}")
    for key in ('wte', 'lm_head'):
        print(f"   {key:8s} " + "  ".join(f"{n}:{fmt(r)}"
              for n, r in zip(names, config['state_dict'][key])))
    if any(b.data for b in config['lm_head_bias']):
        print("   bias     " + "  ".join(f"{n}:{b.data:+g}"
              for n, b in zip(names, config['lm_head_bias'])))
    out = karpathy.generate(config, num_samples=1, temperature=0.01, seed=1,
                            verbose=False, logit_model=embed_only)
    print(f"   generates {out[0]}\n")

# One dimension, four tokens. A B C D sit at 1 2 3 4 on the line. Each token's logit
# is a straight line in x; the winner is the upper envelope, switching at 1.5/2.5/3.5.
show(embedding_only('ABCD',
                    mte     = [[1.], [2.], [3.], [4.], [4.]],   # BOS shares D's point, so BOS -> A
                    lm_head = [[4.], [1.], [2.], [3.], [0.]],   # slopes
                    bias    = [-7.5, 0., -1.5, -4., -9.]),      # intercepts; BOS can never win
     "1-D, four tokens (affine head)")

# Two dimensions, no bias. The same cycle needs a second channel: put the tokens 90
# degrees apart and make each lm_head row its predecessor, so the dot product peaks
# on the successor.
ring = [(1., 0.), (0., 1.), (-1., 0.), (0., -1.)]               # A B C D
show(embedding_only('ABCD',
                    mte     = [list(ring[i])           for i in range(4)] + [[0., -1.]],
                    lm_head = [list(ring[(i - 1) % 4]) for i in range(4)] + [[0., 0.]]),
     "2-D, four tokens (no bias)")
```

Neither model was trained. The matrices were chosen to make a point, which is the
opposite of what §4.8 does — but it is the same machinery, and `generate` cannot tell
the difference. What training buys is not the mechanism; it is the arithmetic of
finding numbers like these when nobody knows what they should be.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```
