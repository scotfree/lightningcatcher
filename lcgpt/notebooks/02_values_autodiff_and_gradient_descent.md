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

# 02 — Values: Autodiff and Gradient Descent

Covers `karpathy.py` lines 19–61: the `Value` class, and the box that notebook 01
asked you to accept without opening.

None of this is specific to language models. `Value` is a general reverse-mode
automatic differentiator — swap the transformer for anything built out of the same
operations and it would work unchanged. It is the layer that makes `loss.backward()`
on line 224 a single line in notebook 03.

The reason it comes second: this is the classical machine learning the rest of the
project sits on. Every notebook after this one treats a gradient as something that
arrives; this is where it comes from, and the last two sections use nothing but
`Value` — no tokens, no model, no training loop.

Code cells reproduce the source verbatim, dedented where a fragment sits inside a
class. Anything that is *not* from the source is marked as such.

+++

## 2.1 The `Value` node — lines 19–26

Not a number — a node in a graph. Four fields: the scalar itself, a gradient slot
that starts at zero, the nodes this one was computed from, and the derivative of
this node with respect to each of them.

`_local_grads` is the design decision worth noticing. Most implementations of this
store a closure per node — a little function that knows how to propagate. Here the
local derivative is computed during the *forward* pass and stored as a plain
number, so `backward` is arithmetic over data rather than a chain of function
calls.

`__slots__` avoids a per-instance `__dict__`. A single training step on the default
model builds hundreds of thousands of these.

## 2.2 Standard operations — lines 28–34


Addition and multiplication, each recording what it was built from and what its
local derivatives are.

For $c = a + b$ the derivatives are $\partial c/\partial a = 1$ and
$\partial c/\partial b = 1$ — hence `(1, 1)`. Addition passes gradient through
unchanged, which is exactly why the residual connections of notebook 04 will keep
the backward path short.

For $c = a \cdot b$ they are $\partial c/\partial a = b$ and
$\partial c/\partial b = a$ — hence `(other.data, self.data)`, each factor's
derivative being the *other* factor.

The `isinstance` line lets a bare float appear on either side of an operator by
wrapping it in a `Value` with no children, which makes it a leaf the graph simply
stops at.

## 2.3 Operators derived from the primitives — lines 40–46

Subtraction, division and negation add no new derivative rules. They are rewrites
into `+` and `*`:

$$
-a = a \cdot (-1)
\qquad
a - b = a + (-b)
\qquad
a / b = a \cdot b^{-1}
$$

So `a / b` builds two nodes — a `__pow__` and a `__mul__` — and the chain rule
handles the quotient rule for free. The `__r*__` variants catch the case where a
plain float is on the left, as in `(1 / n) * sum(losses)` on line 221.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))

def __neg__(self): return self * -1
def __radd__(self, other): return self + other
def __sub__(self, other): return self + (-other)
def __rsub__(self, other): return other + (-self)
def __rmul__(self, other): return self * other
def __truediv__(self, other): return self * other**-1
def __rtruediv__(self, other): return other * self**-1
```

## 2.4 Unary operations — lines 36–39

Four one-liners, each following the same pattern: the forward value, a
single-element children tuple, and the local derivative.

$$
\frac{d}{dx}x^{n} = n x^{n-1}
\qquad
\frac{d}{dx}\log x = \frac{1}{x}
\qquad
\frac{d}{dx}e^{x} = e^{x}
\qquad
\frac{d}{dx}\max(0, x) = [\,x > 0\,]
$$

These four plus `+` and `*` are the entire vocabulary of the model. `log` appears
in the cross-entropy loss, `exp` in the softmax, `relu` in the MLP, and `__pow__`
in `rmsnorm` and in Adam's square root.

ReLU's derivative at exactly zero is a convention — `float(self.data > 0)` picks 0.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

## 2.5 `backward()` — lines 48–61

Two phases.

**Topological sort.** A depth-first walk from the root, appending each node *after*
its children, with a `visited` set so a node reached by several paths is emitted
once. The result is an order in which every node appears after everything it was
built from.

**The reverse sweep.** Seed the root with $\partial \mathcal{L}/\partial \mathcal{L} = 1$,
walk the order backwards, and push each node's gradient down to its children:

$$
\frac{\partial \mathcal{L}}{\partial c} \;\mathrel{+}=\;
\frac{\partial v}{\partial c} \cdot \frac{\partial \mathcal{L}}{\partial v}
$$

That single line is the chain rule. Reversed topological order is what makes it
correct in one pass: when a node is processed, every path from the loss into it has
already been accounted for, so its `grad` is final before it is used.

The `+=` matters. A node used in several places — a weight touched at every
position of a document — receives a contribution from each, and they sum. It is
also why `train` must reset `p.grad = 0` on line 235: nothing here clears anything.

Note that `build_topo` recurses, so graph depth is bounded by Python's recursion
limit. Depth grows with document length, not with corpus size.

```{code-cell} ipython3
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

## 2.6 Derivatives for free

Not from the source. Everything above is the whole mechanism; this is what it buys,
on a function with no model anywhere near it — a cubic and its slope.

$$
f(x) = 2x^{3} - 3x^{2} - 12x + 5
\qquad
f'(x) = 6x^{2} - 6x - 12 = 6(x+1)(x-2)
$$

`f` is written once, in ordinary Python, with `Value` in place of `float`.
Evaluating it builds a graph; `backward()` walks that graph and leaves $f'(x)$ in
`x.grad`. No derivative of $f$ was ever coded — only the derivatives of $x^{n}$,
$+$ and $\times$ were, in §2.2 and §2.4, and the chain rule of §2.5 assembled the
rest. The analytic $f'$ is plotted over the top as a check; the two agree to
floating-point round-off (~1e-15, printed by the cell).

The dotted verticals sit at $x = -1$ and $x = 2$, the roots of $f'$. Where the
derivative crosses zero, $f$ is flat — which is the entire fact the next section,
and all of training, runs on.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Not from the source: f(x) = 2x^3 - 3x^2 - 12x + 5, differentiated by the graph itself.
import sys; sys.path.insert(0, '..')   # karpathy.py lives one level up
import matplotlib.pyplot as plt
from karpathy import Value

def f(x):
    return 2 * x**3 - 3 * x**2 - 12 * x + 5

xs, ys, dys = [i / 20 for i in range(-60, 81)], [], []   # x from -3.0 to 4.0
for xv in xs:
    x = Value(xv)      # a fresh leaf each time, so grad starts at 0
    y = f(x)           # forward: evaluate f *and* build the graph
    y.backward()       # backward: fill in df/dx
    ys.append(y.data)
    dys.append(x.grad)

analytic = [6 * xv**2 - 6 * xv - 12 for xv in xs]
print(f"max |autodiff - analytic| = {max(abs(a - b) for a, b in zip(dys, analytic)):.1e}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.axhline(0, lw=1, color='0.75', zorder=0)
for root in (-1.0, 2.0):                       # f'(x) = 0
    ax.axvline(root, lw=1, ls=':', color='0.75', zorder=0)
    ax.plot(root, f(Value(root)).data, 'o', ms=7, color='#2a78d6', zorder=3)
ax.plot(xs, ys,       lw=2, color='#2a78d6', label="f(x)")
ax.plot(xs, dys,      lw=2, color='#eb6834', label="f'(x), from x.grad")
ax.plot(xs, analytic, lw=1, ls='--', color='0.25', label="f'(x), analytic")
ax.set_xlabel('x')
ax.set_title("$f(x) = 2x^3 - 3x^2 - 12x + 5$, and its derivative for free", color='0.15')
ax.legend(frameon=False, loc='upper left')
ax.tick_params(colors='0.35')
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color('0.75')
plt.show()
```

## 2.7 Gradient descent, on its own

Not from the source. The smallest thing that exercises everything above: a scalar
function with a known minimum, descended by hand.

$$
f(x) = (x-3)^2
\qquad
f'(x) = 2(x-3)
$$

We'll use this computational machinery for this simple example, just to see it on its own before getting into the complexities that the GPT model uses it for.

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
lr = 0.1

for step in range(30):
    y = (x - 3.0) ** 2   # forward: build a fresh graph
    x.grad = 0           # clear last step's gradient
    y.backward()         # backward: fill in dy/dx
    x.data -= lr * x.grad
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d} | x {x.data:+.5f} | f(x) {y.data:.6f} | df/dx {x.grad:+.5f}")
```

```{code-cell} ipython3

```
