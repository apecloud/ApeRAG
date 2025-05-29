import base64
from hashlib import md5
from typing import Any

from aperag.docparser.base import AssetBinPart, CodePart, ImagePart, MarkdownPart, Part, TextPart, TitlePart
from aperag.docparser.parse_md import extract_data_uri, parse_md


def test_parse_md_empty_input():
    input_md = ""
    metadata: dict[str, Any] = {}
    expected_parts: list[Part] = [MarkdownPart(markdown="", metadata={})]
    actual_parts = parse_md(input_md, metadata)
    assert actual_parts == expected_parts


def test_parse_md_simple_text():
    input_md = "This is a simple text."
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert actual_parts[0].markdown == input_md
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == input_md


def test_parse_md_title():
    input_md = "# This is a title\n## level 2"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 3
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TitlePart)
    assert actual_parts[1].content == "# This is a title"
    assert actual_parts[1].level == 1
    assert isinstance(actual_parts[2], TitlePart)
    assert actual_parts[2].content == "## level 2"
    assert actual_parts[2].level == 2


def test_parse_md_lheading_title():
    input_md = "This is a title\n===============\n\nlevel 2\n-------"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 3
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TitlePart)
    assert actual_parts[1].content == "# This is a title"
    assert actual_parts[1].level == 1
    assert isinstance(actual_parts[2], TitlePart)
    assert actual_parts[2].content == "## level 2"
    assert actual_parts[2].level == 2


def test_parse_md_code_block():
    input_md = "    print('Hello, world!')\n"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], CodePart)
    assert actual_parts[1].content == "```\nprint('Hello, world!')\n```"
    assert actual_parts[1].lang is None


def test_parse_md_fence_code_block():
    input_md = "```python\nprint('Hello, world!')\n```"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], CodePart)
    assert actual_parts[1].content == "```python\nprint('Hello, world!')\n```"
    assert actual_parts[1].lang == "python"


def test_parse_md_blockquote_single_line():
    input_md = "> This is a blockquote."
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    # Expected: MarkdownPart, TextPart (blockquote content)
    assert len(actual_parts) == 2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == "> This is a blockquote."


def test_parse_md_blockquote_multi_line():
    input_md = "> This is a blockquote.\n> It has multiple lines."
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    # Expected: MarkdownPart, TextPart (blockquote content)
    assert len(actual_parts) == 2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == "> This is a blockquote.\n> It has multiple lines."


def test_parse_md_blockquote_with_heading():
    input_md = "> # Blockquoted Heading\n> Some text."
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    # Expected: MarkdownPart, TitlePart (blockquoted), TextPart (blockquoted)
    assert len(actual_parts) == 3
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TitlePart)
    assert actual_parts[1].content == "> # Blockquoted Heading"
    assert isinstance(actual_parts[2], TextPart)
    assert actual_parts[2].content == "> Some text."


def test_parse_md_nested_blockquote():
    input_md = "> This is an outer blockquote.\n> > This is a nested blockquote."
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 3
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == "> This is an outer blockquote."
    assert isinstance(actual_parts[2], TextPart)
    assert actual_parts[2].content == "> > This is a nested blockquote."


def test_parse_md_blockquote_with_code():
    input_md = "> ```python\n> print('Hello from blockquote!')\n> ```"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], CodePart)  # Assuming code blocks are CodeParts now
    assert actual_parts[1].content == "```python\nprint('Hello from blockquote!')\n```"
    # Note: The "> " prefixes are not part of the code block content itself
    # They are handled by the blockquote parsing, so we check the code content


def test_parse_md_blockquote_with_image():
    image_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    encoded_data = base64.b64encode(image_data).decode("utf-8")
    mime_type = "image/png"
    data_uri = f"data:{mime_type};base64,{encoded_data}"
    asset_id = md5(image_data).hexdigest()
    input_md = f'> An image: ![the image alt text]({data_uri} "the title")'
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    # Expected: MarkdownPart, AssetBinPart, TextPart(containing image ref)
    assert len(actual_parts) == 4
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], AssetBinPart)
    assert actual_parts[1].asset_id == asset_id
    assert actual_parts[1].mime_type == mime_type
    assert actual_parts[1].data == image_data

    assert isinstance(actual_parts[2], TextPart)
    # The text part should now contain the asset URL
    assert (
        actual_parts[2].content
        == f'> An image: ![the image alt text](asset://{asset_id}?mime_type=image%2Fpng "the title")'
    )

    assert isinstance(actual_parts[3], ImagePart)
    assert actual_parts[3].url == f"asset://{asset_id}?mime_type=image%2Fpng"
    assert actual_parts[3].alt_text == "the image alt text"
    assert actual_parts[3].title == "the title"


def test_parse_md_hr():
    input_md = "---"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == "----"


def test_parse_md_html_block():
    input_md = "<h1>hello world</h1><p>html block</p>"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == input_md


def test_parse_md_simple_ordered_list():
    input_md = "1. First item\n2. Second item"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 3  # MarkdownPart, TextPart for item1, TextPart for item2
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == "1. First item"
    assert isinstance(actual_parts[2], TextPart)
    assert actual_parts[2].content == "2. Second item"


def test_parse_md_simple_unordered_list():
    input_md = "- First item\n* Second item"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 3
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == "- First item"
    assert isinstance(actual_parts[2], TextPart)
    assert actual_parts[2].content == "* Second item"


def test_parse_md_list_item_multi_paragraph():
    input_md = "1. First item,\nsecond line\n\n   Second paragraph of first item."
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    # Expecting MarkdownPart, and then TextParts for each paragraph within the list item
    assert len(actual_parts) == 3
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == "1. First item,\n   second line"
    assert isinstance(actual_parts[2], TextPart)
    # The indentation for the second paragraph is important
    assert actual_parts[2].content == "    Second paragraph of first item."


def test_parse_md_list_item_with_image():
    input_md = "1. First item\n\n   Second paragraph contains ![img](http://abc)."
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 4
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    assert actual_parts[1].content == "1. First item"
    assert isinstance(actual_parts[2], TextPart)
    assert actual_parts[2].content == "    Second paragraph contains ![img](http://abc)."
    assert isinstance(actual_parts[3], ImagePart)
    assert actual_parts[3].url == "http://abc"


def test_parse_md_nested_ordered_list():
    input_md = "1. Outer item 1\n   1. Inner item 1.1\n   2. Inner item 1.2\n2. Outer item 2"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 5  # MarkdownPart, Text (Outer1), Text (Inner1.1), Text (Inner1.2), Text (Outer2)
    assert isinstance(actual_parts[0], MarkdownPart)
    assert actual_parts[1].content == "1. Outer item 1"
    assert actual_parts[2].content == "    1. Inner item 1.1"
    assert actual_parts[3].content == "    2. Inner item 1.2"
    assert actual_parts[4].content == "2. Outer item 2"


def test_parse_md_nested_unordered_list():
    input_md = "- Outer item 1\n  * Inner item 1.1\n  * Inner item 1.2\n- Outer item 2"
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 5
    assert isinstance(actual_parts[0], MarkdownPart)
    assert actual_parts[1].content == "- Outer item 1"
    assert actual_parts[2].content == "    * Inner item 1.1"
    assert actual_parts[3].content == "    * Inner item 1.2"
    assert actual_parts[4].content == "- Outer item 2"


def test_parse_md_nested_mixed_list():
    input_md = (
        "1. Outer ordered\n   - Inner unordered 1\n   - Inner unordered 2\n* Outer unordered\n  1. Inner ordered 1"
    )
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 6
    assert isinstance(actual_parts[0], MarkdownPart)
    assert actual_parts[1].content == "1. Outer ordered"
    assert actual_parts[2].content == "    - Inner unordered 1"
    assert actual_parts[3].content == "    - Inner unordered 2"
    assert actual_parts[4].content == "* Outer unordered"
    assert actual_parts[5].content == "    1. Inner ordered 1"


def test_parse_md_definitions():
    input_md = '[label1]: url1 (title1)\n[label2]: url2 "title2"'
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)
    assert len(actual_parts) == 3
    assert isinstance(actual_parts[0], MarkdownPart)
    assert actual_parts[1].content == "[label1]: url1 (title1)"
    assert actual_parts[2].content == "[label2]: url2 (title2)"


def test_parse_md_simple_table():
    input_md = """
| Header 1 | Header 2 |
| -------- | -------- |
| Cell 1.1 | Cell 1.2 |
| Cell 2.1 | Cell 2.2 |
"""
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md.strip(), metadata)

    assert len(actual_parts) == 2  # MarkdownPart, TextPart for the table
    assert isinstance(actual_parts[0], MarkdownPart)
    assert isinstance(actual_parts[1], TextPart)
    expected_table_md = """| Header 1 | Header 2 |
|---|---|
| Cell 1.1 | Cell 1.2 |
| Cell 2.1 | Cell 2.2 |"""
    assert actual_parts[1].content.strip() == expected_table_md.strip()


def test_parse_md_table_with_inline_markdown_and_image():
    image_data = b"testimagedata"
    encoded_data = base64.b64encode(image_data).decode("utf-8")
    mime_type = "image/png"
    data_uri = f"data:{mime_type};base64,{encoded_data}"
    asset_id = md5(image_data).hexdigest()

    input_md = f"""
| Format   | Example                       |
| -------- | ----------------------------- |
| Bold     | **Strong text** \\|           |
| Image    | ![alt text]({data_uri} "title") |
"""
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md.strip(), metadata)

    assert len(actual_parts) == 4  # MarkdownPart, AssetBinPart, TextPart (table), ImagePart
    assert isinstance(actual_parts[0], MarkdownPart)
    assert any(isinstance(p, AssetBinPart) and p.asset_id == asset_id for p in actual_parts)
    assert any(isinstance(p, ImagePart) and p.url == f"asset://{asset_id}?mime_type=image%2Fpng" for p in actual_parts)

    table_part = next(p for p in actual_parts if isinstance(p, TextPart))
    expected_table_md = f"""| Format | Example |
|---|---|
| Bold | **Strong text** \\| |
| Image | ![alt text](asset://{asset_id}?mime_type=image%2Fpng "title") |"""
    assert table_part.content.strip() == expected_table_md.strip()


def test_extract_data_uri_no_data_uri():
    text = "This text has no data URI."
    metadata: dict[str, Any] = {}
    modified_text, asset_parts = extract_data_uri(text, metadata)
    assert modified_text == text
    assert len(asset_parts) == 0


def test_extract_data_uri_single_image():
    image_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    encoded_data = base64.b64encode(image_data).decode("utf-8")
    mime_type = "image/png"
    data_uri = f"data:{mime_type};base64,{encoded_data}"
    text = f"![alt text]({data_uri})"
    metadata: dict[str, Any] = {}
    modified_text, asset_parts = extract_data_uri(text, metadata)
    assert len(asset_parts) == 1
    assert isinstance(asset_parts[0], AssetBinPart)
    asset_id = md5(image_data).hexdigest()
    assert asset_parts[0].asset_id == asset_id
    assert asset_parts[0].mime_type == mime_type
    assert asset_parts[0].data == image_data
    assert modified_text == f"![alt text](asset://{asset_id}?mime_type=image%2Fpng)"


def test_parse_md_complex_document():
    # Data URIs for images used in the document
    smiley_image_data_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    dog_icon_data_b64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

    smiley_asset_id = md5(base64.b64decode(smiley_image_data_b64)).hexdigest()
    dog_asset_id = md5(base64.b64decode(dog_icon_data_b64)).hexdigest()

    input_md = f"""# Document Title (Level 1)

This is the first paragraph with some **bold text** and *italic text*.
It also includes a [link to OpenAI](https://openai.com).

## Section 1: Lists and Blockquotes (Level 2)

Here's an ordered list:
1. First item.
   - Nested unordered item 1.1
     ```python
     # Code block inside nested list
     print("Hello from nested list code block")
     ```
   - Nested unordered item 1.2 with an image: ![Smiley](data:image/png;base64,{smiley_image_data_b64} "A tiny smiley")
2. Second item of the ordered list.
   > This is a blockquote within a list item.
   > It can span multiple lines.
   > > And even have nested blockquotes!
3. Third item.

And an unordered list:
- Unordered item A.
- Unordered item B, with a
  multi-paragraph content.

  This is the second paragraph of item B.
- Unordered item C.

> This is a top-level blockquote.
> It contains a table:
>
> | Animal | Sound | Image in Table |
> | ------ | ----- | -------------- |
> | Dog    | Woof  | ![Dog Icon](data:image/gif;base64,{dog_icon_data_b64} "Dog") |
> | Cat    | Meow  |                |
>
> And some more text after the table within the blockquote.

---

## Section 2: Code, Tables, and More (Level 2)

An indented code block:

    def greet(name):
        return f"Hello, {{name}}!"

A fenced code block with a language:

```javascript
function sayHi() {{
  console.log('Hi there!');
}}
```

### Subsection 2.1: A More Complex Table (Level 3)

| Feature         | Status      | Notes                                     |
| :-------------- | :---------: | :---------------------------------------- |
| Data URI Images | Implemented | Replaced with `asset://` URLs.            |
| Nested Lists    | Supported   | Both ordered and unordered.               |
| Blockquotes     | Supported   | Including nesting and content like tables.|
| Escaped Chars   | `\\|`        | Pipes `\\|` and newlines `<br>` in cells.  |

Another paragraph after the complex table.

[ref_label]: https://www.example.com "Reference Title"

End of the document.
"""
    metadata: dict[str, Any] = {}
    actual_parts = parse_md(input_md, metadata)

    # Basic checks - you'll want to add more specific assertions
    assert len(actual_parts) > 10  # Expecting many parts
    assert isinstance(actual_parts[0], MarkdownPart)
    assert actual_parts[0].markdown.startswith("# Document Title (Level 1)")  # Check modified MD

    # Check for AssetBinParts
    asset_ids_found = {part.asset_id for part in actual_parts if isinstance(part, AssetBinPart)}
    assert smiley_asset_id in asset_ids_found
    assert dog_asset_id in asset_ids_found

    # Check for ImageParts
    image_urls_found = {part.url for part in actual_parts if isinstance(part, ImagePart)}
    assert f"asset://{smiley_asset_id}?mime_type=image%2Fpng" in image_urls_found
    assert f"asset://{dog_asset_id}?mime_type=image%2Fgif" in image_urls_found

    # Check for a specific title
    titles = [part for part in actual_parts if isinstance(part, TitlePart)]
    assert any(title.content == "# Document Title (Level 1)" and title.level == 1 for title in titles)
    assert any(
        title.content == "## Section 1: Lists and Blockquotes (Level 2)" and title.level == 2 for title in titles
    )
    assert any(
        title.content == "### Subsection 2.1: A More Complex Table (Level 3)" and title.level == 3 for title in titles
    )

    # Check for a specific code block
    code_blocks = [part for part in actual_parts if isinstance(part, CodePart)]
    assert any('print("Hello from nested list code block")' in cb.content and cb.lang == "python" for cb in code_blocks)
    assert any("console.log('Hi there!');" in cb.content and cb.lang == "javascript" for cb in code_blocks)
    assert any("def greet(name):" in cb.content and cb.lang is None for cb in code_blocks)  # Indented

    # Check for a specific table (as TextPart)
    text_parts = [part for part in actual_parts if isinstance(part, TextPart)]
    assert any("| Animal | Sound | Image in Table |" in tp.content for tp in text_parts)
    assert any("![Dog Icon]" in tp.content for tp in text_parts)  # Image in table
    assert any("| Feature | Status | Notes |" in tp.content for tp in text_parts)

    # Check for a specific list item text
    assert any("1. First item." in tp.content for tp in text_parts)
    assert any("    - Nested unordered item 1.1" in tp.content for tp in text_parts)  # Check indentation
    assert any("    - Nested unordered item 1.2 with an image: ![Smiley]" in tp.content for tp in text_parts)
    assert any("    > > And even have nested blockquotes!" in tp.content for tp in text_parts)  # Nested blockquote

    # Check for definition
    assert any(tp.content == "[ref_label]: https://www.example.com (Reference Title)" for tp in text_parts)
