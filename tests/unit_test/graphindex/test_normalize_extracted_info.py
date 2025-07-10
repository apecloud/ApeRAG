"""
Comprehensive unit tests for normalize_extracted_info function.

This module tests all normalization rules including:
1. Chinese parentheses normalization
2. Chinese dash normalization
3. Space removal between Chinese characters
4. Space removal between Chinese and English/numbers
5. Quote removal from start/end
6. Chinese quote removal (for entities)
7. English quote removal around Chinese (for entities)
8. English entity title case normalization (for entities)
"""

from aperag.graph.lightrag.utils import normalize_extracted_info


class TestNormalizeExtractedInfo:
    """Comprehensive tests for normalize_extracted_info function."""

    def test_chinese_parentheses_normalization(self):
        """Test replacement of Chinese parentheses with English parentheses."""
        test_cases = [
            ("中国（北京）", "中国(北京)"),
            ("公司（有限责任）", "公司(有限责任)"),
            ("测试（一）和（二）", "测试(一)和(二)"),
            ("（开始）内容（结束）", "(开始)内容(结束)"),
            # Mixed cases
            ("中文（中）and English (eng)", "中文(中)and English (eng)"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=False)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_chinese_dash_normalization(self):
        """Test replacement of Chinese dashes with English dashes."""
        test_cases = [
            ("北京—上海", "北京-上海"),
            ("2020年—2021年", "2020年-2021年"),
            ("开始－结束", "开始-结束"),
            ("中国—美国—欧洲", "中国-美国-欧洲"),
            # Mixed with English
            ("Chinese—English-mix", "Chinese-English-mix"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=False)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_space_removal_between_chinese_characters(self):
        """Test removal of spaces between Chinese characters."""
        test_cases = [
            ("中 国", "中国"),
            ("北 京 大 学", "北京大学"),
            ("人 工 智 能", "人工智能"),
            ("中   国", "中国"),  # Multiple spaces
            ("中\t国", "中国"),  # Tab character
            ("中\n国", "中国"),  # Newline character
            # Actually removes spaces between Chinese and English
            ("中国 and 美国", "中国and美国"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=False)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_space_removal_between_chinese_and_english(self):
        """Test removal of spaces between Chinese and English/numbers/symbols."""
        test_cases = [
            ("中国 AI", "中国AI"),
            ("AI 技术", "AI技术"),
            ("北京 123", "北京123"),
            ("123 号", "123号"),
            ("中国 ()", "中国()"),
            ("[] 中文", "[]中文"),
            ("中文 @", "中文@"),
            ("# 标签", "#标签"),
            ("$ 符号", "$符号"),
            ("% 百分比", "%百分比"),
            ("! 感叹号", "!感叹号"),
            ("& 符号", "&符号"),
            ("* 星号", "*星号"),
            ("- 减号", "-减号"),
            ("= 等号", "=等号"),
            ("+ 加号", "+加号"),
            ("_ 下划线", "_下划线"),
            # Multiple spaces
            ("中国   AI", "中国AI"),
            ("AI   技术", "AI技术"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=False)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_english_quote_removal_from_start_end(self):
        """Test removal of English quotes from start and end."""
        test_cases = [
            ('"hello world"', "hello world"),
            ("'hello world'", "hello world"),
            ('"中国"', "中国"),
            ("'中国'", "中国"),
            ('"mixed 中英文"', "mixed中英文"),
            # Should not remove if only one side
            ('"hello world', '"hello world'),
            ('hello world"', 'hello world"'),
            ("'hello world", "'hello world"),
            ("hello world'", "hello world'"),
            # Should not remove inner quotes
            ('"hello "world""', 'hello "world"'),
            # Empty quotes
            ('""', ""),
            ("''", ""),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=False)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_chinese_quote_removal_for_entities(self):
        """Test removal of Chinese quotes for entities only."""
        # Updated: The function now properly processes Unicode Chinese quotes
        test_cases = [
            # Unicode Chinese quotes are now processed by the function
            ("\u201c中国\u201d", "中国"),
            ("\u201c人工智能\u201d", "人工智能"),
            # Unicode Chinese single quotes are now processed
            ("\u2018中文\u2019", "中文"),
            ("\u2018测试\u2019", "测试"),
            # Mixed quotes - all Chinese quotes are removed
            ("\u201c开始\u201d和\u201c结束\u201d", "开始和结束"),
            ("\u2018第一\u2019和\u2018第二\u2019", "第一和第二"),
        ]

        for input_text, expected in test_cases:
            # Should only work for entities
            result_entity = normalize_extracted_info(input_text, is_entity=True)
            result_non_entity = normalize_extracted_info(input_text, is_entity=False)

            assert result_entity == expected, (
                f"Entity: '{input_text}' -> Expected: '{expected}', Got: '{result_entity}'"
            )
            # Non-entities should preserve the original text
            assert result_non_entity == input_text, (
                f"Non-entity: '{input_text}' -> Expected: '{input_text}', Got: '{result_non_entity}'"
            )

    def test_english_quote_removal_around_chinese_for_entities(self):
        """Test removal of English quotes around Chinese characters for entities."""
        test_cases = [
            ('"中国人工智能', "中国人工智能"),
            ('中国人工智能"', "中国人工智能"),
            ("'中文测试", "中文测试"),
            ("中文测试'", "中文测试"),
            ('""中国', "中国"),
            ("\"'中国", "中国"),
            ("中国'\"", "中国"),
            ('中国""', "中国"),
            # Should not affect quotes not adjacent to Chinese
            ('"hello" 中国', '"hello" 中国'),
            ('中国 "world"', '中国 "world"'),
        ]

        for input_text, expected in test_cases:
            # Should only work for entities
            result_entity = normalize_extracted_info(input_text, is_entity=True)
            assert result_entity == expected, (
                f"Entity: '{input_text}' -> Expected: '{expected}', Got: '{result_entity}'"
            )

    def test_english_entity_title_case_normalization(self):
        """Test title case normalization for English entities."""
        test_cases = [
            # Basic words
            ("apple", "Apple"),
            ("APPLE", "Apple"),
            ("microsoft", "Microsoft"),
            ("MICROSOFT", "Microsoft"),
            # Multiple words
            ("artificial intelligence", "Artificial Intelligence"),
            ("ARTIFICIAL INTELLIGENCE", "Artificial Intelligence"),
            ("new york", "New York"),
            ("united states", "United States"),
            # Preserve acronyms (4 chars or less and all caps)
            ("AI", "AI"),
            ("US", "US"),
            ("EU", "EU"),
            ("USA", "USA"),
            ("USSR", "USSR"),
            # Preserve acronyms with &
            ("AT&T", "AT&T"),
            ("S&P", "S&P"),
            ("R&D", "R&D"),
            # Hyphenated words
            ("short-term", "Short-Term"),
            ("long-term", "Long-Term"),
            ("state-of-the-art", "State-Of-The-Art"),
            ("well-known", "Well-Known"),
            ("self-driving", "Self-Driving"),
            # Articles and prepositions (lowercased when not first)
            ("university of california", "University of California"),
            ("bank of america", "Bank of America"),
            ("ministry of education", "Ministry of Education"),
            ("department of defense", "Department of Defense"),
            ("institute for technology", "Institute for Technology"),
            ("center with research", "Center with Research"),
            ("organization by committee", "Organization by Committee"),
            # Mixed with numbers
            ("covid-19", "Covid-19"),
            ("5g technology", "5g Technology"),
            ("windows 10", "Windows 10"),
            ("iphone 13", "Iphone 13"),
            # Special characters
            ("johnson & johnson", "Johnson & Johnson"),
            ("procter & gamble", "Procter & Gamble"),
            ("coca-cola", "Coca-Cola"),
            ("t-mobile", "T-Mobile"),
            # Complex cases
            ("new york stock exchange", "New York Stock Exchange"),
            ("federal reserve bank", "Federal Reserve Bank"),
            ("international business machines", "International Business Machines"),
        ]

        for input_text, expected in test_cases:
            # Should only work for entities with English-only content
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_mixed_language_entities_no_title_case(self):
        """Test that mixed language entities are not title-cased."""
        test_cases = [
            ("北京AI研究所", "北京AI研究所"),
            ("中国Microsoft", "中国Microsoft"),
            ("上海Tesla工厂", "上海Tesla工厂"),
            ("AI人工智能", "AI人工智能"),
            ("google中国", "google中国"),
            ("苹果iPhone", "苹果iPhone"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_non_english_entities_no_title_case(self):
        """Test that non-English entities are not affected by title case."""
        test_cases = [
            # Chinese
            ("北京", "北京"),
            ("人工智能", "人工智能"),
            ("中国科学院", "中国科学院"),
            # Should not be affected by case normalization
            ("北京大学", "北京大学"),
            ("清华大学", "清华大学"),
            # Non-ASCII characters that aren't Chinese
            ("Москва", "Москва"),  # Russian
            ("東京", "東京"),  # Japanese
            ("한국", "한국"),  # Korean
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_non_entity_normalization(self):
        """Test that non-entity text doesn't get title case normalization."""
        test_cases = [
            ("this is a description", "this is a description"),
            ("artificial intelligence description", "artificial intelligence description"),
            ("short-term investment analysis", "short-term investment analysis"),
            ("THE COMPANY PROVIDES services", "THE COMPANY PROVIDES services"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=False)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_combined_normalization_rules(self):
        """Test combination of multiple normalization rules."""
        test_cases = [
            # Chinese parentheses + space removal
            ("中 国（北 京）", "中国(北京)"),
            # Chinese dash + space removal
            ("北 京—上 海", "北京-上海"),
            # Quote removal + title case (entity)
            ('"artificial intelligence"', "Artificial Intelligence"),
            # Multiple rules for entities
            ('"中 国 AI"', "中国AI"),
            # Complex entity with multiple rules (mixed language, so no title case)
            ('"SHORT-TERM INVESTMENTS（中 国）"', "SHORT-TERM INVESTMENTS(中国)"),
            # All rules combined (outer quotes removed, internal Chinese spaces processed)
            ('  "  中 国 （ 北 京 ） AI — 技 术  "  ', "中国(北京) AI -技术"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        test_cases = [
            # Empty string
            ("", ""),
            # Only spaces (empty spaces become empty)
            ("   ", ""),
            # Only quotes
            ('""', ""),
            ("''", ""),
            # Single character
            ("a", "A"),  # Entity case
            ("中", "中"),
            # Only punctuation
            ("()", "()"),
            ("-", "-"),
            ("&", "&"),
            # Only numbers
            ("123", "123"),
            ("2020", "2020"),
            # Mixed punctuation and spaces
            ("( ) - &", "( ) - &"),
            # Unicode edge cases
            ("\u4e00", "\u4e00"),  # First Chinese character
            ("\u9fa5", "\u9fa5"),  # Last Chinese character in basic range
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_performance_with_long_text(self):
        """Test performance with long text inputs."""
        # Create a long text with various normalization needs
        long_text = (
            "中 国 人 工 智 能 技 术 " * 100 + "artificial intelligence technology " * 100 + "（测 试）—测 试 " * 100
        )

        # Should not crash or timeout
        result = normalize_extracted_info(long_text, is_entity=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalization_idempotency(self):
        """Test that applying normalization twice gives the same result."""
        test_cases = [
            '"中 国 AI—技 术"',
            "SHORT-TERM INVESTMENTS",
            '"artificial intelligence"',
            "中国（北京）",
            "microsoft corporation",
        ]

        for input_text in test_cases:
            first_pass = normalize_extracted_info(input_text, is_entity=True)
            second_pass = normalize_extracted_info(first_pass, is_entity=True)
            assert first_pass == second_pass, f"Normalization not idempotent for: '{input_text}'"

    def test_real_world_entity_examples(self):
        """Test with real-world entity examples that might appear in documents."""
        test_cases = [
            # Company names
            ("apple inc.", "Apple Inc."),
            ("MICROSOFT CORPORATION", "Microsoft Corporation"),
            ("tesla, inc.", "Tesla, Inc."),
            ("alphabet inc.", "Alphabet Inc."),
            # Financial terms
            ("short-term investments", "Short-Term Investments"),
            ("long-term debt", "Long-Term Debt"),
            ("cash and cash equivalents", "Cash and Cash Equivalents"),
            ("accounts receivable", "Accounts Receivable"),
            # Geographic locations
            ("new york city", "New York City"),
            ("san francisco", "San Francisco"),
            ("united states of america", "United States of America"),
            ("european union", "European Union"),
            # Technology terms
            ("artificial intelligence", "Artificial Intelligence"),
            ("machine learning", "Machine Learning"),
            ("natural language processing", "Natural Language Processing"),
            ("internet of things", "Internet of Things"),
            # Mixed language (should preserve)
            ("北京AI实验室", "北京AI实验室"),
            ("上海Microsoft分公司", "上海Microsoft分公司"),
            ("深圳Tesla超级工厂", "深圳Tesla超级工厂"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_whitespace_normalization(self):
        """Test various whitespace normalization scenarios."""
        test_cases = [
            # Leading/trailing spaces are now removed by .strip()
            ("  中国  ", "中国"),
            ("  artificial intelligence  ", "Artificial Intelligence"),
            # Mixed whitespace characters - tabs and newlines at edges are removed
            ("\t中国\n", "中国"),
            ("\r\n artificial intelligence \t", "Artificial Intelligence"),
            # Complex whitespace scenarios - only internal spaces between Chinese are removed
            ("  \t  中 国  \n  ", "中国"),
            ("  \t  artificial   intelligence  \n  ", "Artificial Intelligence"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_special_character_handling(self):
        """Test handling of special characters and symbols."""
        test_cases = [
            # Common symbols
            ("中国@符号", "中国@符号"),
            ("AT&T corporation", "AT&T Corporation"),
            ("C++ programming", "C++ programming"),  # Special chars prevent title case
            ("Net# framework", "Net# framework"),  # Special chars prevent title case
            # Brackets prevent title case, but parentheses don't
            ("test[1]", "test[1]"),  # Brackets prevent title case
            ("function(param)", "Function(param)"),  # Only parentheses - partial title case
            ("array{index}", "array{index}"),  # Braces prevent title case
            # URLs and emails - special chars prevent title case
            ("http://example.com", "http://example.com"),
            ("user@domain.com", "user@domain.com"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"

    def test_numeric_and_alphanumeric_entities(self):
        """Test normalization of entities containing numbers."""
        test_cases = [
            # Version numbers
            ("version 2.0", "Version 2.0"),
            ("windows 11", "Windows 11"),
            ("ios 15.4", "Ios 15.4"),
            # Mixed alphanumeric
            ("covid-19", "Covid-19"),
            ("h1n1 virus", "H1n1 Virus"),
            ("5g technology", "5g Technology"),
            # Product codes
            ("model abc123", "Model Abc123"),
            ("serial number xyz789", "Serial Number Xyz789"),
        ]

        for input_text, expected in test_cases:
            result = normalize_extracted_info(input_text, is_entity=True)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"
