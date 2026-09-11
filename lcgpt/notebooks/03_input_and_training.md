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

# 03 — Training



Now the `Value` objects of notebook 02 come into their own. Here they stop being
an isolated demonstration: the forward pass below builds one graph out of them per document,
all the way to the loss, and `backward()` fills in a gradient for every parameter.

We'll use Value to build simple Gradient Descent, then add the slight optimization Kaparthy applies to end up with the `train()` function used for the full GPT model. First we'll use it on some simple examples.

_Notes_

## 3.1 Hyperparametes for training

Hyperparameters out of the config, then the flat list of every parameter in the
model — `params` is what the optimiser walks, and its order is fixed by dict
insertion order so the two Adam buffers can be plain parallel lists.

`m` and `v` are those buffers, one slot per parameter: a running average of the
gradient, and a running average of its square. They start at zero and persist
across steps, which is the whole difference between Adam and plain gradient
descent.

```{code-cell} ipython3
def train(config, docs, num_steps=1000, verbose=True, logit_model=gpt,  **hyper):
    """Train `config` in place on `docs`, one document per step. Returns the config."""
    #settings = dict(TRAIN_DEFAULTS)
    config.update(hyper)
    learning_rate = config['learning_rate']
    beta1, beta2, eps_adam = config['beta1'], config['beta2'], config['eps_adam']

    uchars, BOS = config['uchars'], config['BOS']
    block_size, n_layer, token_type = config['block_size'], config['n_layer'], config['token_type']

    # Let there be Adam, the blessed optimizer and its buffers
    params = [p for mat in config['state_dict'].values() for row in mat for p in row]
    m = [0.0] * len(params) # first moment buffer
    v = [0.0] * len(params) # second moment buffer
```

## The Forward Step

This assigns probabilities to the possible values at the next step in our sequence based on current state. In generation, we used this to choose a value. 

Now in Training a model we will use these probabilities with corpus examples to general loss functions - roughly, how "unlikely" the example we saw was given our model. We want to minimize this loss to make the observed data more likely. When we run through the forward step to get these probabilities we're calculating it all using Value objects, so we are also populating all the derivatives and metadata we need for this optimization. The Backward step collects all this afterwards so we can use it to update our models, then start again.

+++

## 3.2 Calculate Loss as Cross-entropy

The logits become probabilities, and the loss is the negative log of the
probability the model assigned to the token that actually came next:

$$
\mathcal{L} \;=\; -\frac{1}{n} \sum_{t=0}^{n-1} \log p_{\theta}\!\left(x_{t+1} \mid x_{\le t}\right)
$$

Only the target's probability appears. Everything else enters through the
denominator of the softmax, which is what stops the model from simply predicting
every token at once.

A useful reference point: a model that has learned nothing spreads its mass evenly
and scores $\log V$ — about 3.30 for the 27-token names vocabulary. That is the
number the loss starts at in the demo below.

```{code-cell} ipython3
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)
loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

# Backward the loss, calculating the gradients with respect to all model parameters
loss.backward()
```

## 3.3 Example: Training a Polynomial with Gradient descent

We won't be plotting derivatives so much as training complex models where we can't really visualize the space. Let's 
use this computational machinery for gradient descent with our simple example, just to see it on its own before getting into the complexities that the GPT model uses it for.

The graph is rebuilt from scratch each step, exactly as the training loop rebuilds
the model's graph for every document — and `x.grad` is cleared each time, for the
reason §2.5 gives.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Not from the source: minimise f(x) = (x - 3)^2, whose minimum is obviously at x = 3.
import sys; sys.path.insert(0, '..')   # karpathy.py lives one level up
from karpathy import Value

x = Value(-2.0)
lr = 0.01

for step in range(30):
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d} | x {x.data:+.5f} | f(x) {y.data:.6f} | df/dx {x.grad:+.5f}")
    #print(x.data)
    y = f(x)   # forward: build a fresh graph
    x.grad = 0           # clear last step's gradient
    y.backward()         # backward: fill in dy/dx
    x.data += lr * x.grad
    
```

```{code-cell} ipython3
x = Value(1.0)
lr = 0.01

for step in range(30):
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d} | x {x.data:+.5f} | f(x) {y.data:.6f} | df/dx {x.grad:+.5f}")
    #print(x.data)
    y = f(x)   # forward: build a fresh graph
    x.grad = 0           # clear last step's gradient
    y.backward()         # backward: fill in dy/dx
    x.data += lr * x.grad
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## 3.4 The Adam update — lines 226–235

As above, plain gradient descent would be `p.data -= lr * p.grad`. We'll modify that - Adaptive Moment Estimation ("ADAM") keeps two running
averages instead, and divides the first by the square root of the second:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}}
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}
\qquad
\theta_t = \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The effect is a per-parameter step size: a parameter with consistently small
gradients still moves, because the division rescales it. The $1-\beta^t$ terms
correct for the buffers starting at zero, and matter only in the first few steps.

`lr_t` decays linearly to zero across the run, and `p.grad = 0` clears the
gradient for the next step — the graph is rebuilt from scratch each document, but
the gradients accumulate with `+=` and would otherwise carry over.

```{code-cell} ipython3
# Adam optimizer update: update the model parameters based on the corresponding gradients
# Note this is pure ML, no GPT/etc?
lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
for i, p in enumerate(params):
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
    p.grad = 0
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Example: Train the Bigram Model

Let's return to the (still simple, non-GPT) model we ended the last notebook with. We've seen how to just build a model by vounting n-grams - let's apply this new training infrastructure to do it. 

Not from the source. Notebook 01 built a model out of counting. This one has the
same shape — one row of logits per current token — but the numbers are `Value`
objects that training is allowed to move:

```python
config['state_dict'] = {'bigram': [[Value(0.0) for _ in range(V)] for _ in range(V)]}
```

That is the whole model: a $V \times V$ table, no attention, no embeddings, no
layers. It satisfies `logit_model`, so `train` accepts it unmodified — the param
flattening on line 196 walks a `state_dict` of any shape.

Starting every entry at `0.0` means the model begins perfectly uniform, so the
loss starts at $\log 27 \approx 3.30$.

There is a fact worth checking at the end. Cross-entropy is minimised, for a model
this shape, exactly when its distribution equals the corpus's own conditional
frequencies. So the table it *learns* by gradient descent should converge to the
table notebook 01 built by *counting*:

$$
\operatorname{softmax}(\theta_a)_b \;\longrightarrow\; \frac{C(a, b)}{\sum_j C(a, j)}
$$

Nothing in the training loop knows about counting. The agreement is what the
gradient is for.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Not from the source: the smallest model with parameters in it.
import sys; sys.path.insert(0, '..')   # karpathy.py and lcgpt.py live one level up

from collections import Counter
import karpathy
from karpathy import Value, new_tokenizer, train, generate, softmax
import lcgpt

docs = lcgpt.load_docs_textfile('../data/names.txt', num_docs=500, seed=42, verbose=False)

# a tokenizer with no model attached to it at all
config = new_tokenizer(docs, n_layer=0, learning_rate=0.1)
uchars, BOS, V = config['uchars'], config['BOS'], config['vocab_size']
print("new_tokenizer gives:", sorted(config))
print("...and no state_dict:", 'state_dict' not in config, "\n")

# the entire model
config['state_dict'] = {'bigram': [[Value(0.0) for _ in range(V)] for _ in range(V)]}

def bigram(config, token_id, pos_id, keys, values):
    return config['state_dict']['bigram'][token_id]

train(config, docs, num_steps=4000, logit_model=bigram, verbose=False)

# the same table, counted rather than learned
counts = Counter()
for d in docs:
    seq = [BOS] + [uchars.index(c) for c in karpathy.doc_to_tokens(d, 'letter')] + [BOS]
    for a, b in zip(seq, seq[1:]):
        counts[(a, b)] += 1

worst = 0.0
for a in range(V):
    total = sum(counts[(a, j)] for j in range(V))
    if total < 50:
        continue
    learned = [p.data for p in softmax(config['state_dict']['bigram'][a])]
    worst = max(worst, max(abs(counts[(a, j)] / total - learned[j]) for j in range(V)))
print(f"learned vs counted, worst probability gap: {worst:.4f}\n")

generate(config, num_samples=8, temperature=0.8, seed=3, logit_model=bigram)
```

## 3.6 Example: Train a Embedding & Generator

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

```{code-cell} ipython3

```
