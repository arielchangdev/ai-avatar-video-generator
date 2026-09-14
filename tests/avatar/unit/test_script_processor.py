"""Unit tests for ScriptProcessor edge cases.

Validates Requirements 3.1, 3.2, 3.3, 3.5:
- 3.1: Script length bounds (1-10000 chars)
- 3.2: Language segmentation (Chinese / English / Unknown)
- 3.3: Contiguous, lossless segmentation
- 3.5: Paragraph break detection at consecutive newlines

Covers task 4.3 edge cases:
- Single character input
- Text at exactly the 10000 character limit
- Text with mixed CJK, Latin, emoji, and special symbols
- Paragraph break detection with various newline patterns
"""

import sys

sys.path.insert(0, ".")

import pytest

from modules.script_processor import ScriptProcessor
from models.schemas import LanguageType


@pytest.fixture
def processor() -> ScriptProcessor:
    return ScriptProcessor()


class TestSingleCharacterInput:
    """Single character inputs are valid and produce exactly one segment."""

    def test_single_chinese_char(self, processor):
        result = processor.process("你")
        assert result.total_chars == 1
        assert len(result.segments) == 1
        seg = result.segments[0]
        assert seg.text == "你"
        assert seg.language == LanguageType.CHINESE
        assert seg.start_index == 0
        assert seg.end_index == 1

    def test_single_latin_char(self, processor):
        result = processor.process("A")
        assert result.total_chars == 1
        assert len(result.segments) == 1
        assert result.segments[0].language == LanguageType.ENGLISH

    def test_single_unknown_char(self, processor):
        result = processor.process("\U0001F600")  # emoji
        assert result.total_chars == 1
        assert len(result.segments) == 1
        assert result.segments[0].language == LanguageType.UNKNOWN


class TestExactLengthBoundary:
    """Text at exactly the 10000-character limit is accepted; 10001 is rejected."""

    def test_exactly_max_length_accepted(self, processor):
        text = "a" * 10000
        result = processor.process(text)
        assert result.total_chars == 10000
        # All same language -> single segment
        assert len(result.segments) == 1
        assert result.segments[0].language == LanguageType.ENGLISH

    def test_exactly_max_length_chinese_accepted(self, processor):
        text = "字" * 10000
        result = processor.process(text)
        assert result.total_chars == 10000
        assert result.segments[0].language == LanguageType.CHINESE

    def test_one_over_max_length_rejected(self, processor):
        text = "a" * 10001
        with pytest.raises(ValueError):
            processor.process(text)

    def test_single_char_minimum_accepted(self, processor):
        result = processor.process("x")
        assert result.total_chars == 1


class TestMixedContent:
    """Mixed CJK, Latin, emoji, and special symbols segment correctly and losslessly."""

    def test_mixed_cjk_latin_emoji(self, processor):
        text = "你好Hello\U0001F600世界"
        result = processor.process(text)

        # Round-trip: lossless reconstruction
        reconstructed = "".join(seg.text for seg in result.segments)
        assert reconstructed == text

        # Expected ordered segmentation
        langs = [(seg.text, seg.language) for seg in result.segments]
        assert langs == [
            ("你好", LanguageType.CHINESE),
            ("Hello", LanguageType.ENGLISH),
            ("\U0001F600", LanguageType.UNKNOWN),
            ("世界", LanguageType.CHINESE),
        ]

    def test_special_symbols_in_latin_range(self, processor):
        # ASCII punctuation is classified as English (Latin/ASCII range)
        text = "Hi! @#$%"
        result = processor.process(text)
        reconstructed = "".join(seg.text for seg in result.segments)
        assert reconstructed == text
        assert all(seg.language == LanguageType.ENGLISH for seg in result.segments)

    def test_segments_are_contiguous(self, processor):
        text = "abc你好123世界!!!"
        result = processor.process(text)
        assert result.segments[0].start_index == 0
        assert result.segments[-1].end_index == len(text)
        for i in range(1, len(result.segments)):
            assert (
                result.segments[i].start_index
                == result.segments[i - 1].end_index
            )

    def test_non_ascii_symbol_is_unknown(self, processor):
        # Miscellaneous symbol outside Latin/CJK ranges
        text = "A\u2600B"  # sun symbol
        result = processor.process(text)
        mid = result.segments[1]
        assert mid.text == "\u2600"
        assert mid.language == LanguageType.UNKNOWN


class TestParagraphBreakPatterns:
    """Paragraph breaks are detected at 2+ consecutive newlines only."""

    def test_single_newline_is_not_a_break(self, processor):
        # Single newline (0x0A) is outside Latin range -> unknown segment, no break
        result = processor.process("第一段\n第二段")
        assert result.paragraph_breaks == []

    def test_double_newline_is_a_break(self, processor):
        text = "第一段\n\n第二段"
        result = processor.process(text)
        assert result.paragraph_breaks == [text.index("\n")]

    def test_triple_newline_is_a_break(self, processor):
        text = "A\n\n\nB"
        result = processor.process(text)
        assert result.paragraph_breaks == [1]

    def test_multiple_paragraph_breaks(self, processor):
        text = "一\n\n二\n\n\n三"
        result = processor.process(text)
        expected = [text.index("\n"), text.index("三") - 3]
        assert result.paragraph_breaks == expected
        assert len(result.paragraph_breaks) == 2

    def test_leading_and_trailing_newlines(self, processor):
        text = "\n\n內容\n\n"
        result = processor.process(text)
        # Breaks at position 0 and after the content
        assert result.paragraph_breaks == [0, len("\n\n內容")]

    def test_no_newlines_no_breaks(self, processor):
        result = processor.process("一段連續文字沒有換行")
        assert result.paragraph_breaks == []
