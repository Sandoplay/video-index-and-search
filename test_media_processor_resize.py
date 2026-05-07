from src.media_processor import MediaProcessor


def test_frame_scale_filter_targets_larger_landscape_side() -> None:
    expected = "scale='if(gte(iw,ih),768,-2)':'if(gte(iw,ih),-2,768)'"
    assert MediaProcessor.frame_scale_filter() == expected


def test_frame_scale_filter_supports_custom_dimension() -> None:
    expected = "scale='if(gte(iw,ih),512,-2)':'if(gte(iw,ih),-2,512)'"
    assert MediaProcessor.frame_scale_filter(512) == expected


def main() -> None:
    tests = [
        test_frame_scale_filter_targets_larger_landscape_side,
        test_frame_scale_filter_supports_custom_dimension,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"\nAll {len(tests)} media processor resize tests passed.")


if __name__ == "__main__":
    main()
