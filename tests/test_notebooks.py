import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aerolex import config
from aerolex.bm25 import BM25Index
from aerolex.store import InMemoryVectorStore
from tests.helpers import corpus

NOTEBOOKS = config.PROJECT_ROOT / "notebooks"


class NotebookTests(unittest.TestCase):
    def test_workshop_bm25_cells_accept_raw_query(self):
        notebook = json.loads((NOTEBOOKS / "workshop.ipynb").read_text())
        chunks = corpus()
        env = {"chunks": chunks, "by_id": {c.chunk_id: c for c in chunks},
               "BM25Index": BM25Index}
        selected = {"9ad415be", "c93edd69", "16c66230", "b4d0b791"}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                if cell["id"] in selected:
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), env)
        self.assertIsInstance(env["query"], str)
        self.assertEqual(env["query_tokens"], ["refund", "cancelled", "flight"])
        self.assertTrue(env["kw_search_chunks"])
        self.assertTrue(all(cid in env["by_id"] for cid, _ in env["kw_search_chunks"]))

    def test_json_syntax_ids_and_script_sync(self):
        for path in NOTEBOOKS.glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            ids = [c["id"] for c in notebook["cells"]]
            self.assertEqual(len(ids), len(set(ids)))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{path}:{cell['id']}", "exec")
        solution = json.loads((NOTEBOOKS / "workshop_solution.ipynb").read_text())
        code = "\n\n".join("".join(c["source"]).rstrip() for c in solution["cells"]
                           if c["cell_type"] == "code") + "\n"
        self.assertEqual((NOTEBOOKS / "_solution_script.txt").read_text(), code)

    def test_feature1_and_changed_generation_paths_offline(self):
        for name in ("workshop.ipynb", "workshop_solution.ipynb"):
            with self.subTest(notebook=name), tempfile.TemporaryDirectory() as tmp:
                notebook = json.loads((NOTEBOOKS / name).read_text())
                env = {"__name__": "__notebook_smoke__"}
                with (patch.object(config, "CHUNKS_CACHE", Path(tmp) / "chunks.json"),
                      patch.object(config, "load_api_key", return_value=None),
                      patch.object(sys, "path", list(sys.path)),
                      contextlib.redirect_stdout(io.StringIO())):
                    for cell in notebook["cells"]:
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), cell["id"], "exec"), env)
                        if cell["id"] == "reference-limits":
                            break
                    self.assertTrue(env["art5"].references)
                    limited = env["expander"].expand([env["art7"]], max_added_chunks=0)
                    self.assertTrue(limited.diagnostics)
                    self.assertEqual(limited.chunks, [env["art7"]])
                    if name == "workshop_solution.ipynb":
                        self.assertTrue(env["fetch_referenced_chunks"](env["art5"], env["by_id"]))
                    else:
                        with self.assertRaises(NotImplementedError):
                            env["fetch_referenced_chunks"](env["art5"], env["by_id"])
                    self._run_generation_cells(notebook, env)

    def _run_generation_cells(self, notebook, env):
        art5 = env["art5"]
        env.update(query="EU261 cancellation compensation",
                   fused=[(art5.chunk_id, 1.0)], ranked_chunks=[(art5.chunk_id, 1.0)])
        reranker = SimpleNamespace(
            rerank=lambda query, candidates, top_k: [(art5.chunk_id, 1.0)])
        with patch("aerolex.rerank.CrossEncoderReranker", return_value=reranker):
            for cell in notebook["cells"]:
                code = "".join(cell["source"])
                if cell["cell_type"] != "code":
                    continue
                if ("baseline_chunks =" in code and "candidates =" in code
                        or "baseline_result = generator.generate" in code
                        or "result = generator.generate" in code
                        or "result = Generator().generate" in code):
                    exec(compile(code, cell["id"], "exec"), env)
            self.assertTrue(env["result"]["offline"])
            self.assertIn("Article 7", " ".join(env["result"]["references"]))
            self.assertGreater(len(env["final_chunks"]), len(env["baseline_chunks"]))

            chunks = env["chunks"]
            matrix = np.eye(len(chunks), dtype=np.float32)
            env["store"] = InMemoryVectorStore.from_chunks(chunks, matrix)
            env["bm25"] = BM25Index([c.chunk_id for c in chunks], [c.searchable_text for c in chunks])
            embedder = SimpleNamespace(
                embed_query=lambda query: matrix[[c.chunk_id for c in chunks].index(art5.chunk_id)])
            env["GeminiEmbedder"] = lambda **kwargs: embedder
            env["key"] = env["KEY"] = "fake-key-never-sent"
            env["mem"] = SimpleNamespace(condense=lambda query: query, add_user=lambda text: None,
                                          add_assistant=lambda text: None)
            env["prof"] = SimpleNamespace(context_block=lambda user: "")
            with patch("aerolex.embeddings.GeminiEmbedder", return_value=embedder):
                full = next(c for c in notebook["cells"] if c["cell_type"] == "code"
                            and "pipe = RagPipeline(" in "".join(c["source"]))
                exec(compile("".join(full["source"]), full["id"], "exec"), env)
            self.assertTrue(env["out"]["offline"])
            self.assertTrue(env["out"]["context_expansion"]["provenance"])
