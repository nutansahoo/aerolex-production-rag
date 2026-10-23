# aerolex-production-rag
# ✈️ AeroLex: Production-Grade RAG Pipeline.  A reference implementation of a production-grade, multi-stage Retrieval-Augmented Generation (RAG) system built from first principles in Python—demonstrated over complex statutory aviation regulations (EU261, US DOT 14 CFR 260) and carrier contracts of carriage.

## Local notebook setup

From the repository root, use an existing Python environment (the workshop
setup uses Python 3.12):

```bash
conda activate rag
python -m pip install -e .
python -m pip check
```

The dependencies in `pyproject.toml` include PDF extraction, BM25, Gemini, local cross-encoder
reranking, and notebook tooling. `sentence-transformers` installs its Torch and
Transformers dependencies. Pip reuses installed versions that satisfy the
requirements. This is a dependency declaration, not an exact-version lockfile.

The `-e` flag installs this checkout in editable mode: `aerolex` is importable
from the selected environment, while Python source edits remain live without
reinstallation. Keep the checkout in place; if you move it, rerun the install.
This workshop uses the editable checkout's adjacent `data/`, `cache/`, and
`.env`; it is not a standalone installation of the corpus.

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey).
Save it privately in a plain-text `.env` file at the repository root:

```dotenv
GEMINI_API_KEY=replace_with_your_key
```

The application loads this file through `python-dotenv`. An exported
`GEMINI_API_KEY` takes precedence. `.env` is ignored by Git; never commit it or
put the key in notebook cells. No other application environment variables are
required for the default data and cache locations.

### VS Code and local imports

Open this repository folder in VS Code, install the Python and Jupyter
extensions, open `notebooks/workshop_solution.ipynb`, and select the `rag`
environment as the notebook kernel.

`aerolex` is the import name of this repository's local package, not a separate
dependency to download from PyPI. The distribution name shown by pip is
`aerolex-production-rag`. Installing it with `python -m pip install -e .` makes
it available regardless of the notebook's working directory; no `%cd` or
`PYTHONPATH` adjustment is needed.

After installation, restart the notebook kernel and run the first setup cell.
If it still reports `ModuleNotFoundError: No module named 'aerolex'`, compare
the Python executable in a temporary notebook cell:

```python
import sys
print(sys.executable)
```

with the executable used for the installation in the terminal:

```bash
python -c "import sys; print(sys.executable)"
```

Both should point to the same environment. Select the matching kernel if they
differ. Remove temporary diagnostic cells before committing notebook changes.

### Before running the remaining cells

The four source PDFs are included in `data/`, and their paths are resolved
relative to the repository rather than a particular machine.

The notebook also expects matching `cache/chunks.json` and
`cache/embeddings.npz` artifacts. They are not included in this checkout.
Before running the vector-store cell in section 3, prepare both caches once.
With `rag` active and your key configured, run:

```bash
python -m aerolex.prepare_cache
```

This explicit preparation command loads all four PDFs, uses the existing
chunking and Gemini embedding helpers, validates matching IDs and vector
dimensions, and saves both artifacts. It sends corpus text to Gemini and
consumes API quota; it may take several minutes because of batching and
rate-limit retries. A key being detected locally does not establish model
access or available quota. Network or VPN restrictions can also prevent API
requests from completing.

For the current corpus, successful output should end with:

```text
Cache verified: (234, 768)
```

The command refuses to run if either cache file already exists, avoiding
accidental overwrites and repeated embedding costs. Errors are surfaced
explicitly. An interrupted or failed API run can still consume quota, and
partial embedding batches are not checkpointed for resumption. If rebuilding
intentionally, move both existing cache files aside first.

After success, run section 3's existing vector-store cell; no notebook code
changes or kernel restart are needed. Subsequent runs load the saved caches.
If you change the source PDFs, chunking settings, embedding model, or vector
dimensions, rebuild both artifacts together before using them.

The first use of the local cross-encoder downloads its public model from
Hugging Face. Later profile and feedback cells write SQLite files under
`cache/`. These generated artifacts are ignored by Git. Live Gemini results
can vary, and the dependency list does not guarantee exact reproduction of
historical notebook output.
