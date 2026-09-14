"""Unit tests for ScriptValidator."""

import pytest

import sys
sys.path.insert(0, '.')

from core.validators.script_validator import ScriptValidator


@pytest.fixture
def validator():
    return ScriptValidator()


class TestScriptValidator:
    """Tests for ScriptValidator.validate()."""

    def test_valid_text_accepted(self, validator):
        result = validator.validate("Hello World")
        assert result.is_valid is True
        assert result.errors == []

    def test_valid_chinese_text(self, validator):
        result = validator.validate("你好世界")
        assert result.is_valid is True

    def test_single_char_accepted(self, validator):
        result = validator.validate("x")
        assert result.is_valid is True

    def test_exactly_max_length_accepted(self, validator):
        result = validator.validate("a" * 10000)
        assert result.is_valid is True

    def test_empty_string_rejected(self, validator):
        result = validator.validate("")
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_exceeds_max_length_rejected(self, validator):
        text = "a" * 10001
        result = validator.validate(text)
        assert result.is_valid is False
        assert "10001" in result.errors[0]

    def test_pure_whitespace_rejected(self, validator):
        result = validator.validate("   \t\n  ")
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_text_with_leading_trailing_whitespace_accepted(self, validator):
        result = validator.validate("  hello  ")
        assert result.is_valid is True

    def test_mixed_content_with_whitespace_accepted(self, validator):
        result = validator.validate("  abc\n\ndef  ")
        assert result.is_valid is True
