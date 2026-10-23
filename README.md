# aerolex-production-rag
# ✈️ AeroLex: Production-Grade RAG Pipeline.  A reference implementation of a production-grade, multi-stage Retrieval-Augmented Generation (RAG) system built from first principles in Python—demonstrated over complex statutory aviation regulations (EU261, US DOT 14 CFR 260) and carrier contracts of carriage.


## Clone the Repo

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
4. Copy the key and keep it private. Do not paste it into a notebook, commit it
   to Git, or share it in screenshots.

Free-tier availability, model access, and rate limits depend on the project,
model, and region. Check [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) before
enabling billing or running a full embedding/evaluation job. Do not assume that
every workshop model is available on the free tier.

### 3. Create the Conda environment and install all packages

Navigate to your copy of the project. Replace this example path if necessary:

```powershell
conda env create -f environment.yml
conda activate aerolex
python -m pip install -e .
```

The environment file installs Python 3.11 and the complete workshop stack. 

The editable install (`pip install -e .`) makes the local `aerolex` package
importable from the notebooks. Do not install only
[pyproject.toml](pyproject.toml): it declares the core library dependencies,
but not the notebook and testing tools. Conda users do not also need to install
[requirements.txt](requirements.txt), which is the alternative pip-based spec.

If you already have an `aerolex` environment, update it instead of recreating it:

```powershell
conda env update -n aerolex -f environment.yml
conda activate aerolex
python -m pip install -e .
```

### 4. Set the Gemini API key

Set `GEMINI_API_KEY` in the same PowerShell terminal that will launch Jupyter:

```powershell
$env:GEMINI_API_KEY = Read-Host "Paste your Gemini API key"
```

The prompt avoids putting the key itself in the command history, but its input
is visible while you paste it. This variable is available to Python and Jupyter
started from this terminal and lasts until the terminal closes.

To also persist it for future terminals on Windows:

```powershell
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $env:GEMINI_API_KEY, "User")
```

New terminals inherit the persisted value. Restart an already-open VS Code
instance if you want its notebook kernels to inherit that value.

**Alternative: a project-local `.env` file.** If you prefer not to persist the
key as a Windows environment variable, create the file without overwriting an
existing one, then edit it:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set its contents to the following, replacing the placeholder with your key:

```dotenv
GEMINI_API_KEY=PASTE_YOUR_API_KEY_HERE
```

Save the file. The project's configuration loader reads this file automatically;
it does not set a global Windows environment variable. The file is already
excluded by [.gitignore](.gitignore). An existing process environment variable
takes precedence over the `.env` value, so keep them consistent or use only one.

### 5. Verify the environment

With `aerolex` activated, run:

```powershell
python --version
python -m pip check
python scripts\check_env.py
python -c "from aerolex.config import load_api_key; key = load_api_key(); print('GEMINI_API_KEY is configured' if key else 'GEMINI_API_KEY is missing'); raise SystemExit(0 if key else 1)"
```

Expect Python 3.11, no broken package requirements, successful package imports,
and four corpus PDFs. The final command checks key presence without printing
the secret; it does not validate the key with Google. The pre-flight script
reports key/cache status separately, so check those lines even if it says
the required checks passed.

To verify live API access, run this small embedding request. It consumes API
quota and may incur a charge depending on your project's billing/tier:

```powershell
python -c "from google import genai; from aerolex.config import load_api_key; client = genai.Client(api_key=load_api_key()); result = client.models.embed_content(model='gemini-embedding-001', contents='Workshop setup check'); print('Gemini embedding request succeeded')"
```

This checks embedding access, not access to every generation model used by the
workshop. For API errors, check the key's project/model access and restrictions;
for `429` errors, check rate limits and available quota.

## Four-variation retrieval evaluation

Open [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) with the `aerolex`
kernel and run its cells. It reuses [notebooks/evaluation.py](notebooks/evaluation.py)
and its 49 questions (including explicit EU context for all 17 European questions).
The default candidate and answer-context sizes are:

| Variation | Retrieval / ranking | Answer context |
|---|---|---|
| Lexical | 20 BM25 candidates | Top 10 lexical chunks |
| Dense | 20 vector candidates | Top 10 dense chunks |
| RRF | Top 20 fused candidates | Top 10 RRF chunks |
| RRF + cross-encoder | Rerank all 20 RRF candidates | Top 10 reranked chunks |

Only the fourth variation uses the cross-encoder. It can promote chunks outside
the RRF top 10; it does not merely reorder the same 10 chunks. The evaluation
explicitly passes its context budget (10 by default) to the reranker rather than
using the pipeline's separate `RERANK_TOP_K = 5` default. Candidate sizes and the
cross-encoder model are configured in [aerolex/config.py](aerolex/config.py).

Every answer uses an independent Gemini Interactions request with
`gemini-3.5-flash-lite`. A full run makes 49 query-embedding calls, 49 local
reranking calls, and 196 answer-generation calls, excluding retries. Existing
generation retries use short waits for `429`/`503`/`UNAVAILABLE`; embedding
retries also recognize rate-limit retry delays. Exhausted retries stop the run.

The notebook defaults to `notebooks\evaluation_results_four_variants.csv`.
Set `LIMIT = 1` for a smoke test, or adjust `CONTEXT_TOP_K` for all four arms.
An existing output is protected unless `OVERWRITE = True`.
Completed four-answer rows are flushed to disk, but partial runs are not
automatically resumed.

The script also runs all four variations:

```powershell
python notebooks\evaluation.py --output notebooks\evaluation_results_four_variants.csv
python notebooks\evaluation.py --limit 1 --output cache\four_variants_smoke.csv
```

Both entry points require `GEMINI_API_KEY`, `cache\chunks.json`, and the
workshop's `cache\new_embeddings.npz` (`gemini-embedding-001`, 768 dimensions).
The script checks dimensions and chunk IDs, but the cache does not record model
provenance. The local cross-encoder loads once per run; its first use may download
weights.

CSV columns include question ID, category, source, question, model, context
budget, and separate answer/context columns:

- `lexical_chunk_ids`, `lexical_answer`
- `dense_chunk_ids`, `dense_answer`
- `rrf_chunk_ids`, `rrf_answer`
- `rrf_reranked_chunk_ids`, `rrf_reranked_answer`

Each ID column is an ordered JSON array of all chunks supplied as context to
that answer. Answers cite actual IDs as `[[chunk_id]]`; full chunk text is not
duplicated in the CSV. Categories and expected sources are output metadata, not
retrieval filters. Equal chunk counts do not imply equal token counts, and a
single run does not control for generation sampling variability.
