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


def test_normalize_segments_limits_to_three_groups():
    analyzer = VisionAnalyzer()
    raw_segments = [
        {
            "start_frame": i,
            "end_frame": i,
            "description": f"part {i}",
        }
        for i in range(6)
    ]

    summary, segments = analyzer._normalize_structured_data(
        {"summary": "summary", "segments": raw_segments},
        make_frames(6),
    )

    assert summary == "summary"
    assert len(segments) == 3
    assert segments[0].start_frame == 0
    assert segments[0].end_frame == 1
    assert segments[1].start_frame == 2
    assert segments[1].end_frame == 3
    assert segments[2].start_frame == 4
    assert segments[2].end_frame == 5


def test_structured_schema_contract():
    schema = VisionAnalyzer.STRUCTURED_ANALYSIS_SCHEMA
    assert schema["required"] == ["summary", "segments"]
    assert schema["additionalProperties"] is False

    segments_schema = schema["properties"]["segments"]
    assert segments_schema["maxItems"] == VisionAnalyzer.MAX_STRUCTURED_SEGMENTS
    assert segments_schema["minItems"] == 1

    segment_schema = segments_schema["items"]
    assert segment_schema["required"] == ["start_frame", "end_frame", "description"]
    assert segment_schema["additionalProperties"] is False


if __name__ == "__main__":
    test_parse_plain_json()
    test_parse_fenced_json()
    test_parse_invalid_json_fails()
    test_normalize_segments_clamps_frame_indices()
    test_normalize_segments_limits_to_three_groups()
    test_structured_schema_contract()
    print("Structured analysis smoke tests passed.")
