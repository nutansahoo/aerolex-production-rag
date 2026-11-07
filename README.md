# aerolex-production-rag
# ✈️ AeroLex: Production-Grade RAG Pipeline.  A reference implementation of a production-grade, multi-stage Retrieval-Augmented Generation (RAG) system built from first principles in Python—demonstrated over complex statutory aviation regulations (EU261, US DOT 14 CFR 260) and carrier contracts of carriage.


## Clone the Repo
```
git clone https://github.com/nutansahoo/aerolex-production-rag.git
```

## Workshop setup (Windows, macOS, and Linux)

Choose local **Conda / venv** below, or the [Google Colab browser-only path](#google-colab-no-local-python-installation).

Complete these steps before the workshop. Run the commands from the project root, the folder containing [environment.yml](environment.yml) and
[pyproject.toml](pyproject.toml).
Unless labeled otherwise, the commands work in PowerShell (Windows) and
Bash or Zsh (macOS / Linux).

### 1. Install prerequisites

- Choose **Conda or venv** in step 3; you do not need both. For Conda, install
  [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
  or Anaconda (it installs Python for you). For venv, install
  [Python 3.11](https://www.python.org/downloads/) with pip first; Conda is not
  required.
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
4. Copy the key and keep it private. Configure it using one of the options in step 4.


### 3. Create an environment and install the project

From the repository root, choose **one** of the following paths.

#### Option A: Conda (all platforms)

Use a Conda-enabled terminal. On Windows, initialize PowerShell by running
`conda init powershell` from Anaconda Prompt. On macOS / Linux, use
`conda init bash` or `conda init zsh` for your shell from a terminal where
Conda is available. Close and reopen the terminal after initialization.

```shell
conda env create -f environment.yml
conda activate aerolex-rag
python -m pip install -e .
```

The environment file installs Python 3.11 and workshop dependencies.
The editable install makes the local `aerolex` package available to notebooks
and installs any remaining dependencies declared in `pyproject.toml`.

#### Option B: Python venv (no Conda required)

**Windows / PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If PowerShell blocks activation, no execution-policy change is required.
Instead, use the environment's Python directly for installation and all later
`python` commands:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

**macOS / Linux (Bash or Zsh)**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If Python 3.11 is installed under a different command, use that command to
create the environment. Some Linux distributions require their Python 3.11
`venv` OS package if environment creation reports it missing.

The venv path installs dependencies from `pyproject.toml`, not the Conda
environment's additional tooling or exact version constraints. `.venv/` is
already ignored by Git.

**For either path:** select the `aerolex-rag` or `.venv` environment's kernel
in VS Code. An existing environment with the required dependencies is also
fine. The Python import remains `aerolex` regardless of environment name.
Continue with the shared steps below.

### 4. Set the Gemini API key

**Option A: local `.env` file (recommended; works on all platforms)**

1. In your editor, create a plain-text file named exactly `.env` in the
   repository root, alongside `pyproject.toml`.
2. Add the following line, replacing the placeholder with your actual key,
   then save:

   ```dotenv
   GEMINI_API_KEY=your_actual_key_here
   ```

The notebook and cache-preparation commands automatically load this file via
`aerolex.config.load_api_key()` and `python-dotenv`; no shell export or notebook
code changes are needed. An existing `GEMINI_API_KEY` environment variable
takes precedence over `.env`. Restart the notebook kernel after changing the
file so it can load the new value.

`.env` is already ignored by Git. NEVER commit it or paste the key into
notebook cells, outputs, or screenshots. From the repository root,
`git check-ignore .env` should print `.env`. Step 5 verifies key presence
without displaying it.

**Option B: environment variable (instead of `.env`)**

**Windows / PowerShell**

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

**macOS / Linux (Bash or Zsh)**

In the terminal that will launch Jupyter, securely enter the key without
putting its value in shell history:

```bash
export GEMINI_API_KEY="$(python -c 'import getpass; print(getpass.getpass("Gemini API key: "))')"
```

This applies only to that terminal session and processes launched from it.
Use Option A for persistent local configuration without shell-profile changes.

### 5. Verify the environment

With your chosen environment active, run (or use its Python executable directly):

```shell
python --version
python -m pip check
python -c "import aerolex; print(aerolex.__file__)"
python -c "from aerolex.config import load_api_key; key = load_api_key(); print('GEMINI_API_KEY is configured' if key else 'GEMINI_API_KEY is missing'); raise SystemExit(0 if key else 1)"
```
Expect the selected Python version, no broken package requirements, and a path
to this checkout's `aerolex` package. The final command checks key presence
without printing it.

## Google Colab (no local Python installation)

[Open the solution notebook in Colab](https://colab.research.google.com/github/nutansahoo/aerolex-production-rag/blob/main/notebooks/workshop_solution.ipynb)

Colab supplies a managed Python/Linux runtime in your browser: no local Conda,
venv activation, VS Code extensions, or kernel registration. Project packages
still need installation in each fresh VM. This route is intended for a standard
**CPU runtime**; a GPU is not required.

### First run

1. Open the link, **save a copy in Drive**, and connect a managed Python runtime
   with no accelerator. Saving a copy preserves your notebook edits, not the VM.
2. In the **Secrets** panel (key icon), add `GEMINI_API_KEY` and enable notebook
   access. Each participant uses their own key and access permission. No `.env`
   upload is needed; never paste credentials into cells, outputs, or clone URLs.
3. Before any `aerolex` imports, run the notebook's **Colab-only setup cell**.
   Opening an ipynb from GitHub does not copy its package, PDFs, or caches into
   the runtime. The cell clones this public repo into
   `/content/aerolex-production-rag`, installs it using the runtime's Python
   (`python -m pip`, equivalent to `%pip` for interpreter selection), and loads
   the Secret into the environment for the existing API-key loader.
4. Keep `COLAB_REF` aligned with the notebook version you opened. `main` is the
   moving default; for an event, instructors should test and publish the same
   commit in the Colab URL (`.../blob/<commit>/notebooks/workshop_solution.ipynb`)
   and `COLAB_REF`. The setup reuses an existing matching checkout but never
   pulls, resets, or overwrites it to change versions. Save your work before
   using a fresh runtime or deliberately managing Git yourself.
5. In the existing setup cell, set **`CHUNKING_STRATEGY = "hierarchical"`** for
   the first run. The public repo includes the four PDFs and hierarchical
   caches; it does **not** include recursive caches. Leave the notebook's other
   settings alone, then run top to bottom. Document vectors load from cache;
   query embeddings and generation still call Gemini.

If dependency installation asks for a session restart, restart and rerun the
Colab setup before the application cells. Do not install Conda or create a
nested venv inside Colab. The local Conda/venv paths and notebook strategy
default are unchanged.

The first reranker use downloads a public model. Colab resources are limited
and not guaranteed, and its preinstalled Python/packages change over time.
**Colab access (including paid Colab plans) does not supply Gemini API quota**;
model access, billing, and rate limits belong to your Gemini project.

### Optional recursive caches and persistence

After the baseline works, run this in a Colab code cell:

```python
!python -m aerolex.prepare_recursive_cache
```

This sends corpus text to Gemini and consumes quota. It refuses to overwrite
existing recursive artifacts; do not rerun blindly after an error. Once it
succeeds, change `CHUNKING_STRATEGY` to `"recursive"` and rerun the tutorial in
order. After a kernel restart, rerun the Colab setup to reload the Secret.

VM packages, cloned files, model downloads, recursive caches, and SQLite state
are ephemeral. A session restart may retain disk, but a deleted/recycled VM
does not. Download or privately back up matching cache pairs and desired state
before disposal. Optional Drive mounting grants notebook code access to Drive;
copy needed artifacts to/from `/content` rather than running all I/O on Drive.
Drive mounting is not required for the recommended setup.

**ZIP alternative:** upload and extract a project ZIP without `.env`, `.venv`,
or secrets, find its root containing `pyproject.toml` and `data/`, and install
that checkout with `%pip install -e /content/path/to/project`. Skip the
GitHub-cloning bootstrap; load the Secret separately before the existing setup:

```python
import os
from google.colab import userdata

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

Uploading only the ipynb does not include its code/data/assets. Review notebook
code before granting Secrets or Drive access; notebook outputs are shareable.

### Instructor pilot checklist

This workflow is based on [Colab's documentation](https://research.google.com/colaboratory/faq.html);
**a full hosted Colab run has not yet been verified for this project**.
Before sharing an event link, try a fresh CPU runtime and confirm:

- The bootstrap completes; `import aerolex; print(aerolex.__file__)` points
  inside `/content/aerolex-production-rag`, and no credential is displayed.
- The hierarchical cache aligns at **234 × 768**, with no document-embedding
  API requests; query embedding, reranking, and the final answer complete.
- The diagrams render. Colab may not resolve notebook-relative SVG links even
  after cloning; the [GitHub assets](notebooks/assets/rag-workshop/) provide a
  fallback. Confirm frontend rendering before changing image URLs or adding
  Colab-specific display code.
- A new runtime can repeat setup; Gemini quota errors are handled as service
  limits rather than by reinstalling environments. Inspect concrete package
  conflicts before adding pins or upgrading unrelated preinstalled packages.

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

With your chosen environment active (or using its Python executable directly),
run from the repository root:

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