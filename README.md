# Causal Models & LLMs: Reviewing Discovered Graphs with Domain Knowledge
**A Practical Series on Combining Causal Discovery Algorithms with Large Language Models**

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Creation date: April 2026
- Last update: May 2026

## Objective

This repository explores **how Large Language Models can complement classical causal discovery algorithms**. Statistical methods return graphs that the data alone cannot fully resolve — edges may be undirected, mis-oriented, or spurious. LLMs carry broad domain knowledge but cannot reason from raw observational data. The two are complementary.

The series covers the main families of causal discovery and pairs each with an LLM integration:

- **Part 1 — Constraint-based** (PC, FCI): post-hoc LLM edge review.
- **Part 2 — Score-based, continuous optimisation, permutation-based** (GES, NOTEARS, GRaSP): LLM knowledge injected *inside* the score / loss.

For a deeper dive into the concepts and implementation details, check out the full article on [The AI Practitioner](https://aipractitioner.substack.com/).

## Project Description

This project demonstrates two ways to combine [`causal-learn`](https://github.com/py-why/causal-learn) / NOTEARS with a provider-agnostic LLM layer (Anthropic, OpenAI, Gemini):

1. **Post-hoc LLM review** — the discovered CPDAG/PAG is passed to the LLM edge-by-edge; each edge is kept, removed, reversed, or oriented in a structured tool call.
2. **LLM-in-the-score** — the LLM produces a penalty matrix (or hard immutability prior) that is folded directly into the local score function used by GES / GRaSP, or into the NOTEARS loss. Domain knowledge then influences the graph *during* search, not after.

### LLM Review Component

The `causal_llm_review` package provides:

- **`decode_adj_matrix`**: turns a `causal-learn` adjacency matrix into typed `EdgeInput` objects with explicit edge marks (`-->`, `---`, `<->`, `o->`, `o-o`).
- **`LLMAdapter`**: a thin abstraction over Anthropic, OpenAI, and Gemini tool/function calling so the same pipeline runs against any provider.
- **`CausalGraphReviewer`**: one structured call per graph that returns an `EdgeDecision` per edge (`keep` / `remove` / `reverse` / `orient`, confidence, reasoning) and applies the decisions back to the adjacency matrix.
- **`EdgePenaltyResponse`**: structured penalty-matrix output used to inject LLM knowledge into score / loss functions.

All outputs are Pydantic-typed and forced through provider tool-use, so they are machine-readable and auditable.

### Code Structure

```
notebooks/
├── constraint_based_algorithms.ipynb   # PC / FCI + post-hoc LLM edge review
└── score_based_algorithms.ipynb        # GES / NOTEARS / GRaSP + LLM-in-the-score

src/causal_llm_review/
├── models.py        # Pydantic schemas: EdgeInput, EdgeDecision, EdgePenaltyResponse
├── graph.py         # decode_adj_matrix: causal-learn matrix → typed edge list
├── adapters.py      # LLMAdapter + Anthropic / OpenAI / Gemini implementations
├── prompts.py       # Jinja2 system + user templates
└── reviewer.py      # CausalGraphReviewer: review() + apply_corrections()

data/
└── adult.csv        # Adult Census Income dataset (illustrative example)
```

## How to Use This Repository?

### Requirements

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

Main libraries:
```
# Causal discovery
causal-learn
notears
pydot

# Data & ML
numpy
pandas
scikit-learn
matplotlib

# LLM providers
anthropic
openai
google-generativeai

# Structured prompting
jinja2
pyyaml
pydantic

# Notebooks
jupyter
ipykernel
ipython
```

### Installation

1. **Install uv** (if not already installed)

2. **Clone the repository**:
```bash
git clone <your-repo-url>
cd causal-models-ai
```

3. **Install dependencies with uv**:
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install the package
uv pip install -e .
```

4. **Install Graphviz** (required by `pydot` to render causal graphs):
```bash
brew install graphviz        # macOS
# or
sudo apt-get install graphviz # Ubuntu/Debian
```

### Setup

1. **Create a `secrets/secrets.yaml` file** with the API keys for the providers you intend to use:
```yaml
anthropic_api_key: sk-ant-your-key-here
openai_api_key:    sk-your-key-here
google_api_key:    your-google-key-here
```

The adapters fall back to the `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GOOGLE_API_KEY` environment variables if the YAML file is missing.

### Running the Project

Two end-to-end notebooks on the Adult Census Income dataset:

**`notebooks/constraint_based_algorithms.ipynb`** — PC (`fisherz` / `chisq`), PC with background knowledge on immutable variables, FCI for latent confounders, and a post-hoc LLM review applied to the FCI output.

**`notebooks/score_based_algorithms.ipynb`** — GES (BIC / BDeu / kernel `CV_general`), NOTEARS continuous optimisation, and GRaSP permutation search. The second half shows how to modify the score function itself: an immutability prior, and an LLM-derived penalty matrix folded into GRaSP's BDeu score and NOTEARS' loss.


### Key Features Demonstrated

- **Discovery algorithms**: PC, FCI (constraint-based); GES with BIC / BDeu / kernel scores (score-based); NOTEARS (continuous optimisation); GRaSP (permutation-based)
- **Two LLM integration patterns**: post-hoc edge review *and* LLM-derived penalty matrices injected into the score / loss
- **Background knowledge**: hard immutability constraints applied either via `BackgroundKnowledge` (PC/FCI) or by penalising forbidden parents inside a custom local score (GES/GRaSP)
- **Provider-agnostic LLM layer**: same pipeline runs against Anthropic, OpenAI, or Gemini
- **Structured LLM output**: Pydantic schemas + tool/function calling for auditable, machine-readable decisions and penalties
- **Round-trip with `causal-learn`**: decisions and custom scores plug into the original adjacency / search loop without mutating internals


## Resources

- **causal-learn Documentation**: https://causal-learn.readthedocs.io/
- **PC & FCI Algorithms (overview)**: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/index.html
- **GES / GRaSP (score- and permutation-based)**: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/index.html
- **NOTEARS**: https://github.com/xunzheng/notears
- **Adult Census Income Dataset**: https://archive.ics.uci.edu/dataset/2/adult
- **uv Package Manager**: https://github.com/astral-sh/uv
- **Anthropic Tool Use**: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling
- **Gemini Function Calling**: https://ai.google.dev/gemini-api/docs/function-calling
