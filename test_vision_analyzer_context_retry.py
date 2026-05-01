import json
from contextlib import contextmanager
from pathlib import Path

import requests

import src.vision_analyzer as vision_module
from src.models import FrameMetadata
from src.vision_analyzer import VisionAnalyzer


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")


@contextmanager
def patched_attr(target, name, value):
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


def make_frames(count):
    return [
        FrameMetadata(path=f"fake/frame_{index:04d}.jpg", timestamp=float(index), index=index)
        for index in range(count)
    ]


def image_count(payload):
    content = payload["messages"][0]["content"]
    return sum(1 for item in content if item.get("type") == "image_url")


def success_response():
    return FakeResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "valid structured summary",
                                "segments": [
                                    {
                                        "start_frame": 0,
                                        "end_frame": 0,
                                        "description": "valid segment",
                                    }
                                ],
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
            "timings": {"prompt_ms": 1000, "predicted_ms": 500},
        },
    )


def context_error(prompt_tokens, context_size=32000):
    return FakeResponse(
        400,
        {
            "error": {
                "code": 400,
                "message": "request exceeds context",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": prompt_tokens,
                "n_ctx": context_size,
            }
        },
    )


def make_analyzer():
    analyzer = VisionAnalyzer()
    analyzer._is_server_ready = lambda: True
    analyzer._image_to_base64 = lambda path: "ZmFrZQ=="
    return analyzer


def test_structured_tries_max_frames_without_preflight_token_count():
    analyzer = make_analyzer()
    analyzer.CONTEXT_SIZE = 6000
    analyzer.RESERVE_TOKENS = 2000
    analyzer.TOKENS_PER_IMAGE = 1000
    posted_counts = []

    def fake_post(url, json, timeout):
        posted_counts.append(image_count(json))
        return success_response()

    with patched_attr(vision_module.requests, "post", fake_post):
        analyzer.analyze_video_structured(make_frames(10), "", batch_size=0)

    assert posted_counts == [10]


def test_structured_context_error_retries_from_real_server_ctx():
    analyzer = make_analyzer()
    posted_counts = []

    def fake_post(url, json, timeout):
        posted_counts.append(image_count(json))
        if len(posted_counts) == 1:
            return context_error(prompt_tokens=46782, context_size=32000)
        return success_response()

    with patched_attr(vision_module.requests, "post", fake_post):
        summary, segments, timings = analyzer.analyze_video_structured(make_frames(44), "", batch_size=50)

    assert posted_counts == [44, 28]
    assert summary == "valid structured summary"
    assert len(segments) == 1
    assert timings["vlm_prompt_tokens"] == 1000


def test_structured_context_error_can_retry_more_than_once():
    analyzer = make_analyzer()
    posted_counts = []

    def fake_post(url, json, timeout):
        count = image_count(json)
        posted_counts.append(count)
        if count == 44:
            return context_error(prompt_tokens=46782, context_size=32000)
        if count == 28:
            return context_error(prompt_tokens=34086, context_size=32000)
        return success_response()

    with patched_attr(vision_module.requests, "post", fake_post):
        analyzer.analyze_video_structured(make_frames(44), "", batch_size=50)

    assert posted_counts == [44, 28, 24]


TESTS = [
    test_structured_tries_max_frames_without_preflight_token_count,
    test_structured_context_error_retries_from_real_server_ctx,
    test_structured_context_error_can_retry_more_than_once,
]


def main():
    failures = []
    for test in TESTS:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"[FAIL] {test.__name__}: {type(exc).__name__}: {exc}")

    if failures:
        raise SystemExit(1)
    print(f"\nAll {len(TESTS)} vision analyzer context retry tests passed.")


if __name__ == "__main__":
    main()
