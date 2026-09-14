"""AI Avatar Video Generator - Script text validator.

Validates user-provided script text for length constraints and content requirements.
"""

from models.schemas import ValidationResult


class ScriptValidator:
    """講稿文字驗證器。

    驗證使用者輸入的講稿文字是否符合長度限制與內容要求：
    - 長度介於 1 至 10000 個字元之間
    - 至少包含一個非空白字元
    """

    MIN_LENGTH = 1
    MAX_LENGTH = 10000

    def validate(self, text: str) -> ValidationResult:
        """驗證講稿文字。

        Args:
            text: 使用者輸入的講稿文字。

        Returns:
            ValidationResult 包含驗證結果與錯誤訊息清單。
        """
        errors: list[str] = []

        # Length check: must be at least 1 character
        if len(text) < self.MIN_LENGTH:
            errors.append("講稿不得為空，至少需輸入 1 個字元")

        # Length check: must not exceed 10000 characters
        if len(text) > self.MAX_LENGTH:
            errors.append(
                f"講稿字數超過上限，最多 {self.MAX_LENGTH} 個字元，"
                f"目前輸入 {len(text)} 個字元"
            )

        # Non-whitespace check: must contain at least one non-whitespace character
        if len(text) >= self.MIN_LENGTH and text.strip() == "":
            errors.append("講稿內容不得僅含空白字元，須包含至少一個非空白字元")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
