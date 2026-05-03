# Causal Models & LLMs: Reviewing Discovered Graphs with Domain Knowledge
**A Practical Series on Combining Causal Discovery Algorithms with Large Language Models**

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Creation date: April 2026
- Last update: May 2026

## Objective

This repository explores **how Large Language Models can complement classical causal discovery algorithms**. Constraint-based methods like PC and FCI return statistically valid but often partially-oriented graphs: edges may be left undirected, mis-oriented, or spurious because the data alone cannot resolve them. LLMs, on the other hand, carry broad domain knowledge but cannot reason from raw observational data.

This first part of the series focuses on **constraint-based causal discovery** (PC, FCI) and introduces a lightweight **LLM review layer** that audits each edge of the discovered graph using domain knowledge — keeping, removing, reversing, or orienting edges with structured, auditable decisions.

For a deeper dive into the concepts and implementation details, check out the full article on [The AI Practitioner](https://aipractitioner.substack.com/).

## Project Description

Constraint-based causal discovery returns a **CPDAG** (PC) or **PAG** (FCI) — equivalence classes that contain edges of the form `-->`, `---`, `<->`, `o->`, `o-o`. These uncertain marks are honest about what the data can identify, but they leave the practitioner with a graph that is hard to act on.

This project demonstrates a two-stage workflow:

1. **Statistical discovery** with [`causal-learn`](https://github.com/py-why/causal-learn) — PC with `fisherz` / `chisq` independence tests, optional background knowledge, and FCI for latent confounders.
2. **LLM review** with a provider-agnostic adapter layer (Anthropic, OpenAI, Gemini) — every edge is reviewed in a single structured tool call and the decisions are applied back to the adjacency matrix.

### Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
│   Dataset    │──▶│  PC / FCI    │──▶│  decode_adj_    │──▶│  LLM Reviewer    │
│  (Adult)     │   │ (causal-learn)│  │  matrix → edges │   │  (tool-call)     │
└──────────────┘   └──────────────┘   └─────────────────┘   └────────┬─────────┘
                                                                     │
                                                            ┌────────▼─────────┐
                                                            │ apply_corrections│
                                                            │  → reviewed adj  │
                                                            └──────────────────┘
```

### LLM Review Component

The `causal_llm_review` package provides:

1. **`decode_adj_matrix`**: turns a `causal-learn` adjacency matrix into typed `EdgeInput` objects with explicit edge marks (`-->`, `---`, `<->`, `o->`, `o-o`).
2. **`LLMAdapter`**: a thin abstraction over Anthropic, OpenAI, and Gemini tool/function calling so the same review pipeline runs against any provider.
3. **`CausalGraphReviewer`**: orchestrates one structured call to the LLM with a Jinja-templated system prompt, returns one `EdgeDecision` per edge (action: `keep` / `remove` / `reverse` / `orient`, confidence, reasoning), and applies the decisions back to the adjacency matrix.

Decisions are structured via Pydantic schemas and forced through provider tool-use, so the output is always machine-readable and auditable.

### Code Structure

```
notebooks/
└── constraint_based_algorithms.ipynb   # End-to-end walkthrough on the Adult dataset

src/causal_llm_review/
├── models.py        # Pydantic schemas: EdgeInput, EdgeDecision, EdgeReviewResponse
├── graph.py         # decode_adj_matrix: causal-learn matrix → typed edge list
├── adapters.py      # LLMAdapter + Anthropic / OpenAI / Gemini implementations
├── prompts.py       # Jinja2 system + user templates for the review call
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

**Start with the notebook**: open the end-to-end walkthrough on the Adult Census Income dataset:
```
notebooks/constraint_based_algorithms.ipynb
```

It covers:
- Loading and preprocessing the dataset
- Running **PC** with `fisherz` (continuous variables)
- Running **PC** with `chisq` (full discretized dataset)
- Running **PC with background knowledge** to fix immutable variables (`sex`, `age`)
- Running **FCI** to allow for latent confounders
- Running the **LLM review** on the FCI output and applying corrections


### Key Features Demonstrated

- **Constraint-Based Discovery**: PC and FCI from `causal-learn` with `fisherz` and `chisq` CI tests
- **Background Knowledge**: forbidding incoming edges on immutable variables (`sex`, `age`)
- **Latent Confounder Handling**: FCI output with `<->` and `o->` edge marks
- **Provider-Agnostic LLM Layer**: a single review pipeline that runs against Anthropic, OpenAI, or Gemini
- **Structured LLM Output**: Pydantic schemas + tool/function calling for auditable, machine-readable decisions
- **Edge-by-Edge Auditability**: every decision carries an action, confidence level, and reasoning grounded in domain knowledge
- **Round-Trip with `causal-learn`**: decisions are applied back to the original adjacency matrix without mutating it


## Resources

- **causal-learn Documentation**: https://causal-learn.readthedocs.io/
- **PC & FCI Algorithms (overview)**: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/index.html
- **Adult Census Income Dataset**: https://archive.ics.uci.edu/dataset/2/adult
- **uv Package Manager**: https://github.com/astral-sh/uv
- **Anthropic Tool Use**: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling
- **Gemini Function Calling**: https://ai.google.dev/gemini-api/docs/function-calling
