"""AI Avatar Video Generator - Script Processor module.

Parses script text to identify language segments (Chinese, English, Unknown)
and paragraph break positions. Designed for efficient processing of mixed-language
scripts up to 10,000 characters within 3 seconds.
"""

from models.schemas import LanguageSegment, LanguageType, ScriptResult, ValidationResult


def _is_cjk(char: str) -> bool:
    """Check if a character is in CJK Unicode ranges.

    Covers CJK Unified Ideographs and common extension blocks.
    """
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)       # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)    # CJK Unified Ideographs Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Unified Ideographs Extension B
        or (0x2A700 <= cp <= 0x2B73F)  # CJK Unified Ideographs Extension C
        or (0x2B740 <= cp <= 0x2B81F)  # CJK Unified Ideographs Extension D
        or (0x2B820 <= cp <= 0x2CEAF)  # CJK Unified Ideographs Extension E
        or (0x2CEB0 <= cp <= 0x2EBEF)  # CJK Unified Ideographs Extension F
        or (0x30000 <= cp <= 0x3134F)  # CJK Unified Ideographs Extension G
        or (0xF900 <= cp <= 0xFAFF)    # CJK Compatibility Ideographs
        or (0x2F800 <= cp <= 0x2FA1F)  # CJK Compatibility Ideographs Supplement
    )


def _is_latin_ascii(char: str) -> bool:
    """Check if a character is Latin/ASCII.

    Covers basic ASCII printable characters plus Latin Extended blocks.
    """
    cp = ord(char)
    return (
        (0x0020 <= cp <= 0x007E)       # Basic ASCII (space through ~)
        or (0x00A0 <= cp <= 0x024F)    # Latin Extended (Latin-1 Supplement, Extended-A, Extended-B)
        or (0x1E00 <= cp <= 0x1EFF)    # Latin Extended Additional
        or (0x2C60 <= cp <= 0x2C7F)    # Latin Extended-C
        or (0xA720 <= cp <= 0xA7FF)    # Latin Extended-D
        or (0x0009 == cp)              # Tab character
    )


def _classify_char(char: str) -> LanguageType:
    """Classify a single character as Chinese, English, or Unknown."""
    if _is_cjk(char):
        return LanguageType.CHINESE
    elif _is_latin_ascii(char):
        return LanguageType.ENGLISH
    else:
        return LanguageType.UNKNOWN


class ScriptProcessor:
    """講稿文字處理模組。

    負責解析講稿文字，辨識語言區段（中文、英文、未識別）
    並偵測段落分隔位置。
    """

    MIN_LENGTH = 1
    MAX_LENGTH = 10000

    def process(self, text: str) -> ScriptResult:
        """解析講稿，辨識語言區段和段落分隔。

        Args:
            text: 使用者輸入的講稿文字。

        Returns:
            ScriptResult containing language segments, paragraph breaks,
            and total character count.

        Raises:
            ValueError: If text is empty, all-whitespace, or exceeds 10000 chars.
        """
        # Validate input
        validation = self._validate(text)
        if not validation.is_valid:
            raise ValueError("; ".join(validation.errors))

        segments = self.detect_language_segments(text)
        paragraph_breaks = self.detect_paragraph_breaks(text)

        return ScriptResult(
            segments=segments,
            paragraph_breaks=paragraph_breaks,
            total_chars=len(text),
        )

    def detect_language_segments(self, text: str) -> list[LanguageSegment]:
        """辨識連續字元區段的語言（中文/英文/未識別）。

        Groups consecutive characters of the same language type into segments.

        Args:
            text: Input text to segment.

        Returns:
            List of LanguageSegment objects preserving original order.
        """
        if not text:
            return []

        segments: list[LanguageSegment] = []
        current_lang = _classify_char(text[0])
        segment_start = 0

        for i in range(1, len(text)):
            char_lang = _classify_char(text[i])
            if char_lang != current_lang:
                # Close the current segment
                segments.append(
                    LanguageSegment(
                        text=text[segment_start:i],
                        language=current_lang,
                        start_index=segment_start,
                        end_index=i,
                    )
                )
                current_lang = char_lang
                segment_start = i

        # Close the final segment
        segments.append(
            LanguageSegment(
                text=text[segment_start:],
                language=current_lang,
                start_index=segment_start,
                end_index=len(text),
            )
        )

        return segments

    def detect_paragraph_breaks(self, text: str) -> list[int]:
        """偵測段落分隔位置（連續換行符號）。

        Identifies positions where 2 or more consecutive newline characters occur.
        Returns the starting index of each paragraph break sequence.

        Args:
            text: Input text to analyze.

        Returns:
            List of character positions where paragraph breaks begin.
        """
        breaks: list[int] = []
        i = 0
        n = len(text)

        while i < n:
            if text[i] == "\n":
                # Count consecutive newlines
                j = i
                while j < n and text[j] == "\n":
                    j += 1
                newline_count = j - i
                if newline_count >= 2:
                    breaks.append(i)
                i = j
            else:
                i += 1

        return breaks

    def _validate(self, text: str) -> ValidationResult:
        """Validate script text meets processing requirements.

        Args:
            text: The script text to validate.

        Returns:
            ValidationResult with validation outcome.
        """
        errors: list[str] = []

        if len(text) < self.MIN_LENGTH:
            errors.append("講稿不得為空，至少需輸入 1 個字元")

        if len(text) > self.MAX_LENGTH:
            errors.append(
                f"講稿字數超過上限，最多 {self.MAX_LENGTH} 個字元，"
                f"目前輸入 {len(text)} 個字元"
            )

        if len(text) >= self.MIN_LENGTH and text.strip() == "":
            errors.append("講稿內容不得僅含空白字元，須包含至少一個非空白字元")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
