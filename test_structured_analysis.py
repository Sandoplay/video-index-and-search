import json

from src.models import FrameMetadata
from src.vision_analyzer import VisionAnalyzer


def make_frames(count=3):
    return [
        FrameMetadata(path=f"frame_{i:04d}.jpg", timestamp=float(i * 2), index=i)
        for i in range(count)
    ]


def test_parse_plain_json():
    analyzer = VisionAnalyzer()
    parsed = analyzer._parse_structured_response('{"summary": "ok", "segments": []}')
    assert parsed["summary"] == "ok"


def test_parse_fenced_json():
    analyzer = VisionAnalyzer()
    parsed = analyzer._parse_structured_response(
        '```json\n{"summary": "ok", "segments": []}\n```'
    )
    assert parsed["summary"] == "ok"


def test_parse_invalid_json_fails():
    analyzer = VisionAnalyzer()
    try:
        analyzer._parse_structured_response("{not json")
    except json.JSONDecodeError:
        return
    raise AssertionError("Invalid JSON should raise JSONDecodeError")


def test_normalize_segments_clamps_frame_indices():
    analyzer = VisionAnalyzer()
    summary, segments = analyzer._normalize_structured_data(
        {
            "summary": "summary",
            "segments": [
                {
                    "start_frame": -10,
                    "end_frame": 99,
                    "description": "full video",
                }
            ],
        },
        make_frames(3),
    )

    assert summary == "summary"
    assert len(segments) == 1
    assert segments[0].start_frame == 0
    assert segments[0].end_frame == 2
    assert segments[0].start_time == 0.0
    assert segments[0].end_time == 4.0


def test_structured_schema_contract():
    schema = VisionAnalyzer.STRUCTURED_ANALYSIS_SCHEMA
    assert schema["required"] == ["summary", "segments"]
    assert schema["additionalProperties"] is False

    segment_schema = schema["properties"]["segments"]["items"]
    assert segment_schema["required"] == ["start_frame", "end_frame", "description"]
    assert segment_schema["additionalProperties"] is False


if __name__ == "__main__":
    test_parse_plain_json()
    test_parse_fenced_json()
    test_parse_invalid_json_fails()
    test_normalize_segments_clamps_frame_indices()
    test_structured_schema_contract()
    print("Structured analysis smoke tests passed.")
