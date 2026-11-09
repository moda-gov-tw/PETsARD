"""Tests for the differentially private LLM text synthesizer (petsard/synthesizer/DP_LLM_synthesizer.py).

差分隱私 LLM 文字合成器的測試。以測試內的假伺服器(只在 127.0.0.1)代替模型，不需要 GPU 或模型檔；
資料是測試內產生的句子。
"""

import importlib.util
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("openai") is None, reason="needs the llm extra (openai)")

WORDS = ["blood", "pressure", "measured", "morning", "evening", "stable", "slightly", "high", "normal", "rest"]


class _FakeModel(BaseHTTPRequestHandler):
    """Answers every request with an invented sentence in the OpenAI Responses format."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, body: bytes, kind: str = "application/json"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(json.dumps({"object": "list", "data": [{"id": "fake", "object": "model"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = '"' + " ".join(random.choices(WORDS, k=9)) + '."'
        item = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}
        resp = {"id": "resp_1", "object": "response", "created_at": int(time.time()), "model": body.get("model", "fake"),
                "status": "completed", "output": [item], "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}
        if body.get("stream"):
            events = [{"type": "response.created", "sequence_number": 0, "response": {**resp, "status": "in_progress", "output": []}},
                      {"type": "response.completed", "sequence_number": 1, "response": resp}]
            self._send("".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode(), "text/event-stream")
        else:
            self._send(json.dumps(resp).encode())


@pytest.fixture(scope="module")
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@pytest.fixture
def texts():
    rng = random.Random(0)
    return pd.DataFrame({"text": [" ".join(rng.choices(WORDS, k=10)) + "." for _ in range(80)]})


def config(url: str, out, **extra) -> dict:
    return {"epsilon": 8, "delta": 1e-5, "epochs": 0, "stop_after_epoch": 0, "num_samples_schedule": 10,
            "sample_num_rows": 10, "api_type": "lmstudio", "api_url": url, "api_key": "lm-studio", "model": "fake",
            "num_procs": 2, "sleep_time": 0, "max_retries": 2, "retry_wait_time": 0,
            "output_dir": str(out / "out"), "log_folder": str(out / "logs"), **extra}


def test_api_type_is_required():
    from petsard.synthesizer.DP_LLM_synthesizer import _create_api

    with pytest.raises(ValueError, match="api_type"):
        _create_api(None, "key", "model")


def test_epsilon_and_delta_are_required():
    from petsard.synthesizer.DP_LLM_synthesizer import DP_LLM_synthesizer

    with pytest.raises(ValueError, match="epsilon"):
        DP_LLM_synthesizer({"delta": 1e-5})
    with pytest.raises(ValueError, match="delta"):
        DP_LLM_synthesizer({"epsilon": 1})


def test_release_profile_then_generate(fake_server, texts, tmp_path):
    from petsard.synthesizer.DP_LLM_synthesizer import DP_LLM_synthesizer

    syn = DP_LLM_synthesizer(config(fake_server, tmp_path))
    syn.fit(texts)
    out = syn.sample()
    assert list(out.columns) == ["text"]
    assert 0 < len(out) <= 10
    report = json.loads((tmp_path / "out" / "privacy_report.json").read_text(encoding="utf-8"))
    assert report["epsilon_spent"] <= report["epsilon_budget"] + 1e-9
    assert report["delta"] == pytest.approx(1e-5)
    assert (tmp_path / "out" / "profile.json").exists()


def test_require_released_profile_stops_without_profile(fake_server, texts, tmp_path):
    from petsard.synthesizer.DP_LLM_synthesizer import DP_LLM_synthesizer

    syn = DP_LLM_synthesizer(config(fake_server, tmp_path, require_released_profile=True))
    syn.fit(texts)
    with pytest.raises(RuntimeError, match="require_released_profile"):
        syn.sample()
