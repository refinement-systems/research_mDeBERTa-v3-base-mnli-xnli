from __future__ import annotations

import email.message
import importlib.util
import io
import json
import pathlib
import sys
import unittest
import urllib.error
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


download_nli_eval_slices = load_module(
    REPO_ROOT / "tools/download-nli-eval-slices.py",
    "download_nli_eval_slices",
)


class FakeJsonResponse(io.StringIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class DownloadNliEvalSlicesTest(unittest.TestCase):
    def test_client_retries_429_using_retry_after(self) -> None:
        headers = email.message.Message()
        headers["Retry-After"] = "7"
        http_error = urllib.error.HTTPError(
            url="https://datasets-server.huggingface.co/rows?dataset=test",
            code=429,
            msg="Too Many Requests",
            hdrs=headers,
            fp=io.BytesIO(b"<html>429</html>"),
        )
        payload = {"rows": [{"row": {"label": 0, "premise": "p", "hypothesis": "h"}}]}
        client = download_nli_eval_slices.DatasetsServerClient(
            "https://datasets-server.huggingface.co",
            request_timeout_seconds=60.0,
            min_request_interval_seconds=0.0,
            max_retries=2,
            initial_backoff_seconds=5.0,
            max_backoff_seconds=120.0,
        )

        with mock.patch("urllib.request.urlopen", side_effect=[http_error, FakeJsonResponse(json.dumps(payload))]):
            with mock.patch("time.sleep") as sleep:
                result = client.fetch_json("/rows", {"dataset": "test"})

        self.assertEqual(result, payload)
        sleep.assert_called_once_with(7.0)

    def test_client_waits_between_request_starts(self) -> None:
        client = download_nli_eval_slices.DatasetsServerClient(
            "https://datasets-server.huggingface.co",
            request_timeout_seconds=60.0,
            min_request_interval_seconds=1.5,
            max_retries=0,
            initial_backoff_seconds=5.0,
            max_backoff_seconds=120.0,
        )
        payload = {"ok": True}
        monotonic_values = iter([10.0, 10.2, 12.0])

        with mock.patch("time.monotonic", side_effect=lambda: next(monotonic_values)):
            with mock.patch("time.sleep") as sleep:
                with mock.patch(
                    "urllib.request.urlopen",
                    side_effect=[
                        FakeJsonResponse(json.dumps(payload)),
                        FakeJsonResponse(json.dumps(payload)),
                    ],
                ):
                    client.fetch_json("/rows", {"dataset": "first"})
                    client.fetch_json("/rows", {"dataset": "second"})

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 1.3)


if __name__ == "__main__":
    unittest.main()
