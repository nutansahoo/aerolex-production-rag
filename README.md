# aerolex-production-rag
# ✈️ AeroLex: Production-Grade RAG Pipeline.  A reference implementation of a production-grade, multi-stage Retrieval-Augmented Generation (RAG) system built from first principles in Python—demonstrated over complex statutory aviation regulations (EU261, US DOT 14 CFR 260) and carrier contracts of carriage.


## Clone the Repo
```
git clone https://github.com/nutansahoo/aerolex-production-rag.git
```

## Workshop setup (Windows / PowerShell)

Complete these steps before the workshop. Run the commands from the project root, the folder containing [environment.yml](environment.yml) and
[pyproject.toml](pyproject.toml).

### 1. Install prerequisites

- Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
  or Anaconda. The workshop environment uses Python 3.11; a separate Python
  installation is not required.
- Use a Conda-enabled PowerShell terminal. If `conda` is not recognized in your
  regular PowerShell terminal, open Anaconda Prompt, run `conda init powershell`,
  then close and reopen PowerShell.
- Download or clone the workshop project and keep its `data` and `cache` folders.
  Internet access is needed for package installation, Gemini API requests, and
  the first download of the local cross-encoder model. A GPU is not required.
- Optional: install [VS Code](https://code.visualstudio.com/) with the Microsoft
  **Python** and **Jupyter** extensions to run the notebooks in VS Code.

### 2. Create a Google account and Gemini API key

1. Create a [Google account](https://accounts.google.com/signup), or use an
   existing account.
2. Open [Google AI Studio](https://aistudio.google.com/), sign in, and complete
   any first-time setup or terms acceptance. This is the developer setup for
   the Gemini API; a paid Gemini chatbot subscription is not required.
3. Open the [API keys page](https://aistudio.google.com/apikey) and choose the
   option to create an API key. Create or select a Google Cloud project if
   prompted; import an existing project into AI Studio if needed.
4. Copy the key and keep it private. Add that key to your environment variable. 


### 3. Create the Conda environment and install all packages

Navigate to your copy of the project and run the following commands:

```powershell
conda env create -f environment.yml
conda activate aerolex
python -m pip install -e .
```

The environment file installs Python 3.11 and the complete workshop stack. 
Create the Conda environment first to install the workshop dependencies. Then run `python -m pip install -e .` to make the local aerolex package available to the notebooks.

### 4. Set the Gemini API key

Set `GEMINI_API_KEY` in the same PowerShell terminal that will launch Jupyter:

```powershell
$env:GEMINI_API_KEY = Read-Host "Paste your Gemini API key"
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $env:GEMINI_API_KEY, "User")
```

OR Follow these steps:

1. Search for System Properties, select Advanced → Environment Variables…
2. Under User variables, click New… and enter:
  - Variable name: GEMINI_API_KEY
  - Variable value: Your API key, without quotes.
3. Click OK on all dialogs, then fully close and reopen VS Code.

### 5. Verify the environment

With `aerolex` activated, run:

```powershell
python --version
python -m pip check
python scripts\check_env.py
python -c "from aerolex.config import load_api_key; key = load_api_key(); print('GEMINI_API_KEY is configured' if key else 'GEMINI_API_KEY is missing'); raise SystemExit(0 if key else 1)"
```
Expect Python 3.11, no broken package requirements, successful package imports,
and four corpus PDFs. The final command checks key presence without printing.

## Comparing chunking strategies

The notebook supports two strategies with the same `Chunk` schema and legal
citations:

| Strategy | Body splitting | Cache files under `cache/` |
| --- | --- | --- |
| `hierarchical` (default) | Existing paragraph packing and character-window fallback | `chunks.json`, `embeddings.npz` |
| `recursive` | LangChain `RecursiveCharacterTextSplitter` within the same legal sections | `chunks_recursive.json`, `embeddings_recursive.npz` |

The recursive splitter tries paragraphs, lines, spaces, then individual
characters. It uses a 1,600-character body limit and 150-character target
overlap. Actual overlap can be smaller; sentence boundaries are not guaranteed.
The heading prepended for retrieval is additional to the body limit. This is
not semantic or token-based chunking, and improved retrieval quality is not
assumed. The original custom strategy and its caches remain unchanged.

### One-time recursive cache preparation

With your existing environment active (`rag` locally or `aerolex` for the
workshop), run from the repository root:

```bash
python -m pip install -e .
python -m aerolex.prepare_recursive_cache
```

The install adds the lightweight `langchain-text-splitters` dependency; the
full LangChain application stack is not required. The preparation command
uses the existing PDF loader and legal-section parser, recursively splits the
original section bodies, and embeds their searchable text with the same Gemini
model and dimensions as the baseline. It validates and saves a matching chunk
and embedding pair, ending with `Cache verified: (N, 768)`. The chunk count
`N` need not match the baseline.

**Preparation sends corpus text to Gemini and consumes API quota.** Configure
`GEMINI_API_KEY` first. The command refuses to overwrite either recursive cache
file and never changes the baseline caches. Failed API requests or interrupted
runs may consume quota; completed batches are not checkpointed. To rebuild
intentionally, move both recursive files aside first. Rebuild the pair whenever
the corpus, splitter settings, or embedding model/dimensions change.

### Select the notebook strategy

In the setup cell of `notebooks/workshop_solution.ipynb`, choose:

```python
CHUNKING_STRATEGY = "recursive"  # "hierarchical" keeps the existing baseline
```

Restart the kernel and run top to bottom after changing this setting. Directly
under the hierarchical example, the new cell defines `alternate_chunks`,
prints a citation example, and selects the downstream `chunks`, lookup
metadata, and embedding-cache path together. If recursive chunks are not
cached, that cell can split them locally without API calls, but section 3
still requires the separately prepared recursive embeddings.

Do not replace `chunks` alone while keeping the baseline vector store: vectors
must correspond to the selected chunk IDs and text. The notebook checks ordered
ID alignment and dimensions before retrieval. BM25 is rebuilt from the selected
chunks; reranking, citations, and the final pipeline use the same selection.
Recursive IDs are namespaced so baseline feedback votes cannot attach to
different recursive passages.

For a comparison, keep PDFs, size settings, embedding model, and retrieval
settings fixed. Chunk counts, boundaries, and overlap behavior will still
differ. Existing profile/feedback state and conversational history can affect
outputs; a clean kernel resets in-memory history, not persisted SQLite state.
This adds a comparison option, not an automated quality evaluation.