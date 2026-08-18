# Mini-GPT — Lightning Catcher Project: State & Context

> **Purpose of this document.** A working context/bootstrap doc for building out a "Lightning Catcher" project around Andrej Karpathy's minimal, dependency-free GPT. It captures the format, the audience, the strategic positioning, the node breakdown, and the detailed decisions made so far — enough to start building notebooks and to continue the design conversation in a fresh session (including Claude Code).
>
> **Not included:** the mathematical content of the separate *Tangent Bundle* project. This is its own, smaller project. (The two are related and may cross-reference later — see §2.)
>
> **The source code** (Karpathy's mini-GPT, ~300 lines of pure Python) is pulled into the workspace separately and can be referenced directly by section/line. It is deliberately **not reproduced here**.

---

## 1. What "Lightning Catcher" is (the format)

**Lightning Catcher** is the name for a general *style* of project, not this specific project:

- **Linked notebook + podcast pairs.** Each node in a project is (roughly) one Jupyter notebook paired with one companion podcast episode.
  - **Notebooks** carry the technical content: clean explanations, real runnable code, math where relevant, illustrations and standalone demos.
  - **Podcasts** carry the human narrative: intuition, motivation, history, "why it's this way, not just how it is," how ideas connect, why something is neat.
- **The animating spirit:** the joy of delving into technical — especially mathematical and computer-science — subjects **purely for the fun of understanding**, with no professional or academic accomplishment required or implied.
- Notebooks are also a quiet demonstration that Jupyter is a lovely, accessible medium for this kind of thing: live code, visualization, nice typesetting, no heavy tooling required to follow along.

### Target audience
Adults (and curious learners generally) who **want to understand hard things for fun** — people who find genuine satisfaction in doing rigorous technical work outside any credential or career context. For this project specifically, the audience is unusually well-primed: **a lot of people already know they're interested in how LLMs/transformers work** and have even tried to read code like this, but admit they get lost in the actual math (and the code implementing it). This project meets exactly that itch.

### Tone / collaboration notes
- Honest pushback is welcome and valued over flattery; comfortable being wrong and working it out live; enjoys deep, exploratory back-and-forth.
- Prose is authored by the project owner (see §9 on authorship division of labor).

---

## 2. This project & its strategic positioning

**This project:** break Karpathy's minimal GPT into digestible, notebook-sized pieces. Each notebook unpacks *what the code actually does* and *lays out the math behind it*, with illustrations and standalone examples so people can put the whole thing together. Roughly **3–10 notebooks** (current plan: **5 core + follow-ups**), each with a companion podcast.

**Why this project, and why now (before the bigger Tangent Bundle project):**
- **Smaller in every sense.** ~5 nodes vs. Tangent Bundle's ~13; concrete and bounded (there's a fixed 300-line artifact anchoring it).
- **More approachable / less grandiose.** Tangent Bundle is a big dream project ("build up to advanced differential geometry from scratch using only the tools we like"). Beautiful, but big, and harder to invite people into — not everyone realizes they'd love to learn measure theory. Almost everyone who's curious about AI already suspects they'd like to understand transformers.
- **Proof of concept for the format.** Doing this one first demonstrates the Lightning Catcher format working end to end, builds trust in the depth, and makes the larger Tangent Bundle project feel inviting rather than intimidating.

**Relationship to Tangent Bundle (genuinely separate, but they talk to each other):**
- This project stands on its own and does **not** depend on any Tangent Bundle content.
- Much of it is naturally "here's what derivatives are *good for*" — gradient descent, optimization — which is exactly the kind of motivation a calculus student appreciates (origin anecdote: discussed with a family member about to take calculus).
- **Later payoff:** near the end of Tangent Bundle, once the manifold machinery exists, one can return here and reframe — a loss landscape as a geometric surface, gradient descent as motion on a manifold, "here's another surface we can do interesting geometry on." A lovely callback, but explicitly *not* a dependency in either direction for now.

**Naming:** project name TBD. (Tangent Bundle is named for its summit; this one could earn a name too — open item.)

---

## 3. The source material

Karpathy's mini-GPT: "the most atomic way to train and run inference for a GPT in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency."

Structural map (the actual file is in the workspace; reference it directly):
- **Data / dataset** — downloads `names.txt` (baby names), one document (name) per line.
- **Tokenizer** — character-level: unique chars → ids `0..n-1`, plus a special **BOS** token; `vocab_size = len(uchars) + 1`.
- **`Value` autograd class** — scalar-valued reverse-mode autodiff: `data`, `grad`, `_children`, `_local_grads`; operator overloads (`+ * ** log exp relu` etc.); `backward()` does a topological sort then walks it in reverse applying the chain rule.
- **Parameters / `state_dict`** — token & position embeddings (`wte`, `wpe`), per-layer attention weights (`wq wk wv wo`), MLP weights (`fc1 fc2`), `lm_head`. Config: `n_layer=1`, `n_embd=16`, `block_size=16`, `n_head=4`, `head_dim=4`.
- **Model forward (`gpt(...)`)** — GPT-2-flavored, with minor swaps: **RMSNorm** (not LayerNorm), **ReLU** (not GeLU), **no biases**. Token+position embedding → attention block (multi-head, causal, KV cache) → MLP block, with residual connections. Helpers: `linear`, `softmax`, `rmsnorm`.
- **Optimizer** — hand-rolled **Adam** with bias correction; linear LR decay; trains one document per step for `num_steps=1000`.
- **Loss** — per-position cross-entropy (`-log p[target]`), averaged over the document.
- **Inference** — samples 20 "hallucinated names" with temperature.

Everything is scalar-valued (no tensors/numpy in the hot path): slow, but maximally legible — every operation is a visible node in the computation graph.

---

## 4. Scope principles (the rules of the game)

1. **Anchor to the 300 lines.** The core nodes explain what's *in* Karpathy's code and *why* it's that way — not a redesign, not sideways extensions, not "here's what a 12-layer model does."
2. **Bite-sized is a feature, not a constraint to fight.** The whole appeal is that the complete algorithm fits in your head. If we expanded every idea into *Principia Mathematica*, we'd lose the thread. Do the **minimum explanation** needed for real understanding.
3. **But context is half the point.** We're not only saying *how* it works — we're saying *why it's like this*. Every node lives on a spectrum between "bare minimum unpacking of the code" and "here's the broader idea this is an instance of." Finding the right point on that spectrum is a per-node judgment call (attention especially — see §5).
4. **Extensions go in follow-up nodes.** Anything not in the code (fine-tuning, deployment, deeper theory) is a *follow-up*, kept in the Lightning Catcher spirit but clearly separated from the core five. This keeps the core tight while leaving room for "now that you understand this, here's what comes next."
5. **Standalone demos are allowed inside a node** when they *illustrate the code's ideas in isolation* (e.g., a tiny gradient-descent demo, an n-gram baseline). The test: does it illuminate what the code is doing, or does it wander off? Illustrating-in-isolation is in-scope; changing Karpathy's code is not.

---

## 5. Node structure

### Core five (faithful to Karpathy's code)

1. **Plumbing & tokenization** — reading the data, building the vocabulary, character↔integer encoding, BOS, dataset stats. *(Detailed in §6 — this is the most developed node so far.)*
2. **Derivatives, computation graphs & gradient descent** — the `Value` class, reverse-mode autodiff (chain rule over a computation graph), the Adam optimizer. The calculus/CS heart. "Here's what derivatives are *for*." Most people have heard of gradient descent — this is where it becomes concrete and *seen*.
3. **The transformer block** — the forward pass as data flow: token + position embeddings, residual connections, RMSNorm, the MLP, how a single token moves through one layer. "Here's what a transformer layer *does*." (The **transformer trick** as its own headline idea.)
4. **Attention** — its own node, deliberately. Multi-head attention, Q/K/V projections, causal masking, softmax as a distribution over the past, the KV cache. "Here's the **attention trick**." This is where most people's intuition genuinely breaks, so it earns dedicated space. (Rich enough that we needn't cover *everything* — pick the spectrum point that illuminates without ballooning.)
5. **Loss & training** — cross-entropy as negative log-likelihood, the training loop, sampling one document at a time, temperature at inference. How the whole thing actually learns.

**Why transformer and attention are split** (they were briefly conflated in discussion): "transformer" is the *architecture* (the whole block — embeddings, attention, MLP, residuals, stacking); "attention" is one mechanism *inside* it. The two ideas people find most distinctive and most confusing are precisely the **transformer trick** and the **attention trick**, so each gets its own node rather than nesting attention inside the block node.

### Follow-up nodes (extend beyond the 300 lines, still Lightning Catcher)
- **Fine-tuning** — taking a trained model and adapting it to new data/tasks; the real-world "vicissitudes of training." Natural next step, but not in the code → follow-up, not squeezed into Node 5.
- **Deployment / serving** — how you actually *use* the thing in the world: APIs, MCP servers ("here's how you'd set up an MCP server to interact with this code"), etc.
- **"Entropy, Markov chains & language"** — a deeper companion to the Shannon sidebar in Node 1 (see §6): Shannon's original method, Markov/n-gram models, and how they relate to what the neural net does end-to-end. Optional deeper dive.
- Others as they emerge.

---

## 6. Node 1 in detail (Plumbing & tokenization)

This node is more fully worked out than the rest. It carries three payloads: the honest plumbing, a key conceptual "tokens are just integers" beat, and a Shannon n-gram baseline.

### 6.1 The plumbing (do it, don't skip it)
- Read `names.txt` (or fetch from the URL); shuffle; count docs.
- Build `uchars` (sorted unique characters); map char↔id; add **BOS**; `vocab_size = len(uchars)+1`.
- Show **encode/decode** on a few example names concretely.
- Dataset stats: number of names, longest name, length distribution — motivates `block_size=16` (note: block size is *longer* than the longest name, so there's always padding room; real systems handle variable length / truncation more carefully).

### 6.2 Conceptual beat: "the model doesn't care what tokens *are*"
A verified, load-bearing point worth foregrounding:
- **Character-level here is a deliberate simplification.** Real LLMs use learned subword/word-level tokenizers (vocab ~50k–100k). For baby names, the minimal sensible tokenization is one character = one token (vocab ~30–40).
- **The architecture is identical regardless.** From the tokenizer onward, *nothing in the code cares* whether a token is a character, a subword, or a word — it's all just integers and the *relationships between them*. Generating a string of characters that follows the trained pattern is the *same operation* as a chatbot constructing a string of words that follows the trained pattern. The only differences are **vocabulary size**, **context-window needs**, and **training efficiency**.
- Pedagogical move: give people a window into "those big scary chat tokens are just integers too; the space is bigger, that's all."

### 6.3 In-node demo A — word-level tokenizer (Thought 1)
Drive 6.2 home by literally reusing Karpathy's code with a different tokenizer:
- The change is ~one line where the tokenizer currently assumes characters; hand it words instead. **Same forward/backward/training loop, unchanged.**
- Point made concretely: "watch — we changed only the tokenizer and everything still runs." This is clarification, not digression → **stays in Node 1**.
- **Open question — data sourcing.** Need a vocabulary *and* a corpus that (a) run in a plain notebook without a GPU (a slow run is fine; not meant to be interactive), yet (b) are interesting *as words*. 500 words is likely already big for CPU-only. Candidates discussed: song lyrics (artist/genre-specific), a single poet's complete works, nursery rhymes / children's stories, niche-subreddit comments, movie/TV scripts, or existing restricted-vocabulary datasets (à la "Goldfish GPT"). Recognizable output (people can feel whether it "sounds right") is a plus — lyrics or poetry lean strongest. **Undecided:** full-convergence demo vs. quick proof-of-concept run. **Undecided:** exact corpus/vocabulary source. *(This may be the thing that pushes the word-level demo toward a follow-on if data turns out to be fiddly — but current lean is it fits in Node 1.)*

### 6.4 In-node demo B — Shannon n-gram baseline (Thought 2)
Once tokenized, we already have everything needed to build a Shannon-style generator in ~10–20 lines:
- A function that chunks the token stream by tuple length `n` and counts n-grams; a function that generates a sequence by sampling from those counts. Run for `n = 1..5`.
- **Expected payoff:** `n=1` is gibberish; by `n=3–4` it looks strikingly name-like — *before any neural network is trained*. Same core idea ("follow the learned distribution over what comes next") at increasing scales of context, exactly as in Shannon's 1948 work.
- **Framing discipline:** do **not** get into entropy theory here. Just: "here's a way to generate from sequence distributions; Shannon did this in the '40s; watch it improve as `n` grows." A sidebar, not a lecture.
- **Use it as a baseline** to compare against the GPT — both on *computational effort* and on *quality as training progresses* (early in training the GPT will likely look *worse* than a high-order n-gram; then catch up and generalize rather than memorize). This makes "why is a neural net better than just counting?" an **empirical** question answered in the notebook.
- Deeper treatment (Markov models, the actual information theory) → the optional follow-up node in §5.

---

## 7. Open questions / TODOs
1. **Word-level demo corpus & vocab** — pick a small-but-interesting dataset that runs CPU-only; decide full-convergence vs. proof-of-concept (§6.3).
2. **Attention node scope** — decide the spectrum point: how much of the "why" (Q/K/V geometry, why softmax, multi-head intuition) to include without ballooning (§5).
3. **Shannon: sidebar vs. follow-on** — confirm the in-Node-1 baseline stays lightweight; scope the optional "Entropy, Markov chains & language" follow-up (§6.4).
4. **Project name** — TBD (§2).
5. **Per-node illustration inventory** — for each node, list the key visualizations/standalone demos (e.g., a computation-graph picture for Node 2; attention-weight heatmaps for Node 4; n-gram-vs-GPT quality-over-training plot for Nodes 1/5).
6. **Follow-up ordering** — which follow-ups to build first (fine-tuning likely leads).

---

## 8. Working surfaces & workflow
- **claude.ai chat surface** — strategy, workshopping, voice-mode conversations. No filesystem / code execution. Can **read** a connected GitHub repo but **not write** to it (and reads may be synced/indexed rather than live — expect sync lag; not a substitute for git diffing branches/commits).
- **Claude Code** — the build surface: reads/writes/runs/commits notebooks directly. This is where notebooks actually get constructed.
- **GitHub repo** — planned, to bridge the two surfaces (voice/strategy here, building in Claude Code).
- **jupytext** — recommended at repo init: pair each `.ipynb` with a plain-text `.py`/`.md` mirror. Gives clean git diffs *and* a paste-friendly representation (raw `.ipynb` JSON is awkward to reason about on the chat surface — escaped strings, `\n`-joined source arrays, cell metadata noise; when discussing a cell there, paste the cell *content*, not the JSON).

---

## 9. Format & authorship conventions (how notebooks get built)
Carried over from the Lightning Catcher way of working:
- **Division of labor.** Notebooks do the technical content cleanly; podcasts carry the human narrative (intuition, motivation, history, "why").
- **Author writes the prose.** Prose/markdown cells are the owner's to write — it's part of the enjoyment. When scaffolding a notebook, provide:
  - **Bullet-point skeletons** for markdown/prose cells (structure to expand from, not finished prose),
  - **Actual LaTeX** for the core math being discussed (a tedious-to-type anchor worth providing),
  - **Real, runnable code** for code cells.
- **Honesty about simplifications.** Where the toy code differs from real systems (char vs. subword tokens; scalar autograd vs. tensors; 1 layer; no biases), say so plainly and point to what "real" looks like — without turning the node into a survey.

---

*End of context document. A fresh session (chat or Claude Code) can begin from here; pull in Karpathy's source file alongside it.*
