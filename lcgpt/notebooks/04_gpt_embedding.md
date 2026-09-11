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

+++

## 3.2 The default configuration — lines 5–17

Every dimension of the model, in one dict. `n_embd` is the width of a token's
vector, `n_layer` the number of stacked blocks, `block_size` the furthest back
attention can look, and `n_head` how many independent attention channels split
that width. The remaining four are the optimiser's.

`block_size=16` is not arbitrary: the longest name in `data/names.txt` is 15
characters, plus one `BOS`.

```{code-cell} ipython3
CONFIG_DEFAULTS = {
    'n_layer': 1,     # depth of the transformer neural network (number of layers)
    'n_embd': 16,     # width of the network (embedding dimension)
    'block_size': 16, # maximum context length of the attention window
    'n_head': 4,      # number of attention heads
    'token_type': 'letter',
    'learning_rate': 0.01, 
    'beta1': 0.85, 
    'beta2': 0.99, 
    'eps_adam': 1e-8
}
```

## 3.4 Tokenizer plus weights — lines 167–183

Not sure this really does anything anymore.

A convenience wrapper: build the vocabulary, then hang a fresh set of weights off
it. `init_params` is notebook 04's subject. Nothing in these notebooks calls this
function — it exists for the command line — but it is the one place that shows the
two halves belong to one object.

The truncation warning matters on an unfamiliar corpus. Documents longer than
`block_size` are silently cut short in the loop below, and on a corpus of song
lyrics rather than names that is most of them.

```{code-cell} ipython3
def new_model_config(docs, token_type='letter', verbose=True, seed=None, **overrides):
    """Build a tokenizer from `docs` and initialise a fresh set of weights."""
    config = new_tokenizer(docs, token_type, **overrides)
    init_params(config, seed)
    if verbose:
        block_size = config['block_size']
        print(f"vocab size: {config['vocab_size']} ({token_type} tokens)")
        print(f"Dimensions: {config['n_layer']} x  {config['n_embd']} x {config['n_embd']} ")
        print(f"Num. Params: {len([p for mat in config['state_dict'].values() for row in mat for p in row])}")
        # Documents longer than the context window are silently cut short by the
        # `n = min(block_size, ...)` in train(). Worth saying out loud on an unfamiliar corpus.
        over = sum(1 for d in docs if len(doc_to_tokens(d, token_type)) + 1 > block_size)
        if over:
            print(f"note: {over} of {len(docs)} documents ({100 * over / len(docs):.0f}%) are longer than "
                  f"block_size={block_size} and will be truncated")
    return config
```

## 3.1 Splitting a document into tokens — lines 149–154

A *token* is whatever unit the model treats as indivisible. Two choices here:

- **`letter`** — one character per token. `list(doc)` and nothing more.
- **`word`** — whitespace-separated, lowercased, with the curly apostrophe folded
  onto the straight one and punctuation trimmed off each end, so that `"Yeah,"`,
  `"yeah"` and `"yeah!"` collapse to a single token rather than three.

The choice changes only how many distinct tokens exist. Nothing downstream can
tell the difference.

```{code-cell} ipython3
def doc_to_tokens(doc, token_type):
    if token_type == 'letter':
        return list(doc)
    WORD_TRIM = '!"\'(),-.?—'
    cleaned = (w.strip(WORD_TRIM) for w in doc.lower().replace('’', "'").split())
    return [w for w in cleaned if w]
```

## 3.3 Building the vocabulary — lines 156–165

Every distinct token in the corpus, sorted, becomes an integer id. Sorting matters
— it makes the mapping reproducible across runs, which is why a saved model can
ship its `uchars` and get the same ids back.

`BOS` takes the one id past the end. It is a real token to the model, and it does
double duty: it starts every sequence and it ends every sequence, so "the document
is finished" is something the model predicts rather than something imposed on it.

```{code-cell} ipython3
def new_tokenizer(docs, token_type='letter', **overrides):
    """Build a config and its tokenizer from `docs`."""
    config = CONFIG_DEFAULTS.copy()
    config.update(overrides)
    config['token_type'] = token_type
    # Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    config['uchars'] = sorted({t for d in docs for t in doc_to_tokens(d, token_type)}) # unique tokens become ids 0..n-1
    config['vocab_size'] = len(config['uchars']) + 1             # +1 is for BOS
    config['BOS'] = len(config['uchars'])                        # id of the Beginning of Sequence token
    return config
```

## 3.6 One document at a time — lines 200–207

No batching. One step, one document, cycling with `%` so the corpus repeats.

The token sequence is wrapped in `BOS` at both ends, then `n` caps the number of
positions at `block_size`. Note what `min` does on a long document: it truncates
silently. The tail is not trained on.

```{code-cell} ipython3
# Repeat in sequence
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    # n determines the number of positions for truncation
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(tok) for tok in doc_to_tokens(doc, token_type)] + [BOS]
    n = min(block_size, len(tokens) - 1)
```

## 3.7 The forward pass — lines 209–216

A fresh KV cache per document, then one call to the model per position. At
position `pos_id` the model sees `tokens[pos_id]` and is asked to predict
`tokens[pos_id + 1]`.

__This is what makes the corpus self-supervising: no labels exist, because every
token is the label for the one before it. A document of length $n$ yields $n$
training signals, not one.__

`logit_model` is the same seam `generate` used — the loop never names the
transformer.

```{code-cell} ipython3
# Forward the token sequence through the model, building up the computation graph all the way to the loss

keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
losses = []
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]

    logits = logit_model(config, token_id, pos_id, keys, values)
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
