"""Property-based test for ScriptProcessor.

Property 5: Script language segmentation round-trip and correctness
For any valid script text (1-10000 chars, at least one non-whitespace), the ScriptProcessor
SHALL produce segments where:
1. Concatenating all segment texts in order reproduces the original text exactly
2. Each segment labeled "zh" contains only CJK characters, each segment labeled "en"
   contains only Latin/ASCII characters, each segment labeled "unknown" contains
   characters outside those ranges
3. Consecutive double-newline positions are identified as paragraph breaks
4. The text is rejected if it is empty, all-whitespace, or exceeds 10000 characters

**Validates: Requirements 3.1, 3.2, 3.3, 3.5**
"""

import sys

sys.path.insert(0, ".")

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    text,
    characters,
    sampled_from,
    composite,
    integers,
    lists,
    one_of,
    just,
)

from modules.script_processor import ScriptProcessor, _is_cjk, _is_latin_ascii
from models.schemas import LanguageType


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# CJK characters (common Chinese characters)
cjk_chars = characters(
    min_codepoint=0x4E00,
    max_codepoint=0x9FFF,
)

# Latin/ASCII printable characters (0x0020-0x007E)
latin_chars = characters(
    min_codepoint=0x0020,
    max_codepoint=0x007E,
)

# "Unknown" characters - characters outside CJK and Latin/ASCII ranges
# Use emoji, Arabic, Cyrillic, etc.
unknown_chars = sampled_from(
    [
        "\u0600",  # Arabic
        "\u0700",  # Syriac
        "\u0E00",  # Thai
        "\u3040",  # Hiragana
        "\u30A0",  # Katakana
        "\U0001F600",  # Emoji
        "\U0001F4A9",  # Emoji
        "\u2600",  # Misc symbol (not in Latin ranges)
        "\n",  # Newline (0x000A) is NOT in Latin range -> unknown
    ]
)


@composite
def mixed_language_text(draw, min_size=1, max_size=200):
    """Generate random mixed-language text with Chinese, English, special chars."""
    segments = draw(
        lists(
            one_of(
                text(alphabet=cjk_chars, min_size=1, max_size=20),
                text(alphabet=latin_chars, min_size=1, max_size=20),
                text(alphabet=unknown_chars, min_size=1, max_size=10),
            ),
            min_size=1,
            max_size=15,
        )
    )
    result = "".join(segments)
    # Ensure non-empty and within bounds
    assume(len(result) >= min_size)
    assume(len(result) <= max_size)
    # Ensure at least one non-whitespace character
    assume(result.strip() != "")
    return result


@composite
def valid_script_text(draw):
    """Generate valid script text: 1-10000 chars, at least one non-whitespace."""
    result = draw(mixed_language_text(min_size=1, max_size=500))
    return result


@composite
def text_with_paragraph_breaks(draw):
    """Generate text that contains paragraph breaks (consecutive double-newlines)."""
    parts = draw(
        lists(
            text(alphabet=cjk_chars, min_size=1, max_size=20),
            min_size=2,
            max_size=5,
        )
    )
    # Join parts with double newlines to create paragraph breaks
    num_newlines = draw(
        lists(
            integers(min_value=2, max_value=4),
            min_size=len(parts) - 1,
            max_size=len(parts) - 1,
        )
    )
    result_parts = []
    for i, part in enumerate(parts):
        result_parts.append(part)
        if i < len(parts) - 1:
            result_parts.append("\n" * num_newlines[i])
    result = "".join(result_parts)
    assume(len(result) <= 10000)
    assume(result.strip() != "")
    return result


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestScriptProcessorProperty:
    """Property 5: Script language segmentation round-trip and correctness.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.5**
    """

    @given(script_text=valid_script_text())
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_segment_concatenation_reproduces_original(self, script_text):
        """Concatenation of all segment texts in order reproduces original text exactly.

        This is the round-trip property: segmentation must be lossless.
        """
        processor = ScriptProcessor()
        result = processor.process(script_text)

        # Round-trip: concatenate all segment texts
        reconstructed = "".join(seg.text for seg in result.segments)
        assert reconstructed == script_text, (
            f"Round-trip failed.\n"
            f"Original ({len(script_text)} chars): {repr(script_text[:100])}\n"
            f"Reconstructed ({len(reconstructed)} chars): {repr(reconstructed[:100])}"
        )

    @given(script_text=valid_script_text())
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_segment_labels_match_character_classification(self, script_text):
        """Each segment labeled 'zh' contains only CJK chars, 'en' only Latin/ASCII,
        'unknown' only characters outside those ranges.
        """
        processor = ScriptProcessor()
        result = processor.process(script_text)

        for seg in result.segments:
            for char in seg.text:
                if seg.language == LanguageType.CHINESE:
                    assert _is_cjk(char), (
                        f"Segment labeled 'zh' contains non-CJK char: {repr(char)} "
                        f"(U+{ord(char):04X}) in segment: {repr(seg.text[:50])}"
                    )
                elif seg.language == LanguageType.ENGLISH:
                    assert _is_latin_ascii(char), (
                        f"Segment labeled 'en' contains non-Latin char: {repr(char)} "
                        f"(U+{ord(char):04X}) in segment: {repr(seg.text[:50])}"
                    )
                elif seg.language == LanguageType.UNKNOWN:
                    assert not _is_cjk(char) and not _is_latin_ascii(char), (
                        f"Segment labeled 'unknown' contains CJK or Latin char: {repr(char)} "
                        f"(U+{ord(char):04X}) in segment: {repr(seg.text[:50])}"
                    )

    @given(script_text=text_with_paragraph_breaks())
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_paragraph_breaks_at_double_newline_positions(self, script_text):
        """Paragraph breaks are correctly identified at consecutive double-newline positions."""
        processor = ScriptProcessor()
        result = processor.process(script_text)

        # Manually find all positions with 2+ consecutive newlines
        expected_breaks = []
        i = 0
        n = len(script_text)
        while i < n:
            if script_text[i] == "\n":
                j = i
                while j < n and script_text[j] == "\n":
                    j += 1
                if j - i >= 2:
                    expected_breaks.append(i)
                i = j
            else:
                i += 1

        assert result.paragraph_breaks == expected_breaks, (
            f"Paragraph breaks mismatch.\n"
            f"Expected: {expected_breaks}\n"
            f"Got: {result.paragraph_breaks}\n"
            f"Text: {repr(script_text[:100])}"
        )

    @given(script_text=valid_script_text())
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_segment_indices_are_contiguous_and_cover_text(self, script_text):
        """Segment start/end indices form a contiguous, non-overlapping partition of the text."""
        processor = ScriptProcessor()
        result = processor.process(script_text)

        assert len(result.segments) > 0, "Valid text must produce at least one segment"

        # First segment starts at 0
        assert result.segments[0].start_index == 0

        # Last segment ends at text length
        assert result.segments[-1].end_index == len(script_text)

        # Segments are contiguous
        for i in range(1, len(result.segments)):
            assert result.segments[i].start_index == result.segments[i - 1].end_index, (
                f"Gap between segment {i-1} (end={result.segments[i-1].end_index}) "
                f"and segment {i} (start={result.segments[i].start_index})"
            )

        # Segment text matches the slice
        for seg in result.segments:
            assert seg.text == script_text[seg.start_index:seg.end_index], (
                f"Segment text doesn't match text slice: "
                f"seg.text={repr(seg.text[:50])}, "
                f"slice={repr(script_text[seg.start_index:seg.end_index][:50])}"
            )

    def test_rejection_for_empty_input(self):
        """Empty string is rejected."""
        processor = ScriptProcessor()
        with pytest.raises(ValueError):
            processor.process("")

    def test_rejection_for_whitespace_only(self):
        """All-whitespace string is rejected."""
        processor = ScriptProcessor()
        with pytest.raises(ValueError):
            processor.process("   \t  ")

    def test_rejection_for_exceeding_max_length(self):
        """Text exceeding 10000 characters is rejected."""
        processor = ScriptProcessor()
        long_text = "a" * 10001
        with pytest.raises(ValueError):
            processor.process(long_text)

    @given(
        whitespace=text(
            alphabet=sampled_from([" ", "\t", "\n", "\r"]),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_rejection_for_any_whitespace_only_input(self, whitespace):
        """Any text containing only whitespace chars is rejected."""
        assume(whitespace.strip() == "")
        processor = ScriptProcessor()
        with pytest.raises(ValueError):
            processor.process(whitespace)

    @given(length=integers(min_value=10001, max_value=15000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_rejection_for_any_overlength_input(self, length):
        """Any text exceeding 10000 characters is rejected regardless of content."""
        long_text = "a" * length
        processor = ScriptProcessor()
        with pytest.raises(ValueError):
            processor.process(long_text)
