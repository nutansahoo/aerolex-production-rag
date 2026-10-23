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