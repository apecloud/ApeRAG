"""
Simplified tests for normalize_extracted_info function to verify core functionality.
"""

from aperag.graph.lightrag.utils import normalize_extracted_info


class TestNormalizeExtractedInfoSimple:
    """Simplified tests for normalize_extracted_info function."""

    def test_basic_chinese_normalization(self):
        """Test basic Chinese text normalization."""
        test_cases = [
            ("中国（北京）", "中国(北京)"),
            ("北京—上海", "北京-上海"),
            ("中 国", "中国"),
            ("中国 AI", "中国AI"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=False)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_english_quote_removal(self):
        """Test English quote removal."""
        test_cases = [
            ('"hello world"', "hello world"),
            ("'hello world'", "hello world"),
            ('"中国"', "中国"),
            ('"mixed中英文"', "mixed中英文"),  # Space removed between mixed content
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=False)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_english_entity_title_case(self):
        """Test English entity title case normalization."""
        test_cases = [
            ("artificial intelligence", "Artificial Intelligence"),
            ("short-term investments", "Short-Term Investments"),
            ("bank of america", "Bank of America"),
            ("AI", "AI"),  # Preserve acronyms
            ("AT&T", "AT&T"),  # Preserve acronyms with &
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_mixed_language_no_title_case(self):
        """Test that mixed language entities don't get title case normalization."""
        test_cases = [
            ("北京AI研究所", "北京AI研究所"),
            ("SHORT-TERM INVESTMENTS中国", "SHORT-TERM INVESTMENTS中国"),
            ("C++ programming", "C++ programming"),  # Special chars prevent title case
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_whitespace_behavior(self):
        """Test how whitespace is handled."""
        test_cases = [
            # Leading/trailing spaces are removed for entity normalization
            ("  中国  ", "中国"),
            ("  artificial intelligence  ", "Artificial Intelligence"),  # Title case and trim spaces
            # Empty space strings become empty
            ("   ", ""),
            # Tab and newline removal between Chinese
            ("中\t国", "中国"),
            ("中\n国", "中国"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_entity_vs_non_entity_behavior(self):
        """Test difference between entity and non-entity normalization."""
        test_cases = [
            # Title case only applies to entities
            ("artificial intelligence", "artificial intelligence", "Artificial Intelligence"),
            # Quote removal works for both
            ('"hello world"', "hello world", "Hello World"),
        ]

        for input_text, expected_non_entity, expected_entity in test_cases:
            result_non_entity = normalize_extracted_info(input_text, is_entity=False)
            result_entity = normalize_extracted_info(input_text, is_entity=True)

            assert result_non_entity == expected_non_entity, (
                f"Non-entity: '{input_text}' -> Expected: '{expected_non_entity}', Got: '{result_non_entity}'"
            )
            assert result_entity == expected_entity, (
                f"Entity: '{input_text}' -> Expected: '{expected_entity}', Got: '{result_entity}'"
            )

    def test_real_world_examples(self):
        """Test with real-world examples from the bug report."""
        test_cases = [
            # These should normalize to consistent forms
            ("short-term investments", "Short-Term Investments"),
            ("Short-term investments", "Short-Term Investments"),
            ("SHORT-TERM INVESTMENTS", "Short-Term Investments"),
            ("Short-Term Investments", "Short-Term Investments"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_complex_combinations(self):
        """Test complex combinations of normalization rules."""
        test_cases = [
            # Quote + space + parentheses + title case
            ('"artificial intelligence"', "Artificial Intelligence"),
            # Chinese + English + quotes + spaces
            ('"中 国 AI"', "中国AI"),
            # Everything combined - realistic expectations for complex input
            ('  "  中 国 （ 北 京 ） AI — 技 术  "  ', "中国(北京) AI -技术"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_edge_cases(self):
        """Test edge cases."""
        test_cases = [
            ("", ""),  # Empty string
            ("a", "A"),  # Single character entity
            ("中", "中"),  # Single Chinese character
            ("123", "123"),  # Numbers only
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"
