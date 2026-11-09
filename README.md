# ✈️ AeroLex: Evidence-first answers for airline passengers

This repo contains the code for building a RAG pipeline for a chatbot for airlines' passenger rights. 

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

1. Search for "View Advanced System Settings" in windows search bar
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
Colab setup before the application cells.

### Optional: recursive caches and persistence
In the setup cell of `notebooks/workshop_solution.ipynb`, choose:

```python
CHUNKING_STRATEGY = "recursive"  # "hierarchical" keeps the existing baseline
```

Restart the kernel and run top to bottom after changing this setting, if you want to try this method of chunking.