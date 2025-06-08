# 文档解析 (`docparser`) 模块

该模块负责解析各种文档格式，提取其内容和结构，并将其准备好在 ApeRAG 系统中进行进一步处理，主要通过将内容分块为可管理的部分。

## 核心功能

`docparser` 模块接收文件路径作为输入，并返回一个结构化的 `Part` 对象列表。这些部分代表文档中不同的语义元素，例如标题、段落、代码块、图像和表格。它通常还会生成文档的完整 Markdown 表示。

---

## 关键组件

1.  **`DocParser` (`doc_parser.py`)**：
    * 作为解析过程的主要入口点和协调器。
    * 管理一个可配置的专业解析器列表（例如，用于音频、图像、Markdown、复杂文档的解析器）。
    * 根据文件扩展名和配置选择合适的解析器。
    * 允许启用/禁用特定解析器并覆盖其设置。

2.  **`BaseParser` 和 `Part` 对象 (`base.py`)**：
    * `BaseParser`：所有独立解析器都必须实现的抽象基类。它定义了一个通用接口，包含 `supported_extensions()` 和 `parse_file()` 等方法。
    * `Part`：一个 Pydantic 模型，表示已解析文档的一个片段。有多种专门的 `Part` 类型：
        * `TextPart`：纯文本内容。
        * `TitlePart`：文档标题，包括其级别。
        * `CodePart`：代码块，包含可选的语言信息。
        * `MarkdownPart`：表示文档的完整 Markdown 内容。
        * `ImagePart`：表示图像，包括其 URL（可能是 `asset://` URL）和替代文本。
        * `MediaPart`：通用媒体部分。
        * `AssetBinPart`：存储嵌入资产（例如从数据 URI 或复杂文档中提取的图像）的二进制数据，以及唯一的 `asset_id` 和 `mime_type`。这些通常由其他部分通过 `asset://<asset_id>` URL 引用。
    * `FallbackError`：解析器无法处理文件并希望允许另一个解析器尝试时引发的异常。

3.  **独立解析器**：
    * **`MarkItDownParser` (`markitdown_parser.py`)**：
        * 处理各种格式，包括 `.txt`、`.md`、`.html`、`.ipynb`、`.pdf`、`.epub` 以及 Microsoft Office 文档（`.docx`、`.doc`、`.xlsx`、`.xls`、`.pptx`、`.ppt`）。
        * 使用 `markitdown` 库进行主要的 Markdown 转换。
        * 对于旧版 Office 格式（`.doc`、`.ppt`），它可以首先使用 `soffice` (LibreOffice/OpenOffice) 将其转换为现代 XML 格式。
        * 生成的 Markdown 随后由 `parse_md.py` 处理。
    * **`DocRayParser` (`docray_parser.py`)**：
        * 专为复杂、布局密集型文档设计，如 `.pdf`、`.docx`、`.doc`、`.pptx`、`.ppt`。
        * 依赖外部“DocRay”微服务（通过 `settings.DOCRAY_HOST` 配置）。
        * 将文档提交给 DocRay，轮询完成状态，并检索结构化的 JSON 输出（“middle_json”）以及提取的图像。
        * 此 `middle_json` 提供详细的布局信息（页面、块、边界框），然后将其转换为 `Part` 对象，包括用于图像的 `AssetBinPart`。
        * 还从这些部分生成合并的 Markdown 表示。
    * **`AudioParser` (`audio_parser.py`)**：
        * 将音频文件（例如，`.mp3`、`.wav`、`.ogg`）转录为文本。
        * 使用外部 Whisper ASR Web 服务（通过 `settings.WHISPER_HOST` 配置）。
        * 输出一个带有转录文本的 `TextPart`。
    * **`ImageParser` (`image_parser.py`)**：
        * 从图像中提取文本 (OCR)，支持 `.jpg`、`.png`、`.bmp` 等格式。
        * 使用外部 PaddleOCR 服务（通过 `settings.PADDLEOCR_HOST` 配置）。
        * 输出一个带有提取文本的 `TextPart`。

4.  **Markdown 处理 (`parse_md.py`)**：
    * 接收 Markdown 字符串（通常是 `MarkItDownParser` 或 `DocRayParser` 的输出），并将其转换为详细的 `Part` 对象列表。
    * 使用 `markdown-it-py` 对 Markdown 进行分词。
    * `PartConverter` 类遍历这些 token，创建相应的 `Part` 对象（例如，`TitlePart`、`TextPart`、`CodePart`、`ImagePart`）。
    * 处理嵌入的 Base64 数据 URI：将其转换为 `AssetBinPart` 对象，并用 `asset://` 链接替换 Markdown 中的 URI。
    * 保留从 Markdown 到 `Part` 对象的源映射（行号）。

5.  **实用函数 (`utils.py`)**：
    * `convert_office_doc()`：使用 `soffice` 在 Office 文档格式之间进行转换。
    * `get_soffice_cmd()`：检查 `soffice` 命令是否存在。
    * `asset_bin_part_to_url()`：为 `AssetBinPart` 创建 `asset://` URL。
    * `extension_to_mime_type()`：将文件扩展名映射到 MIME 类型。

6.  **分块 (`chunking.py`)**：
    * 这个关键组件接收解析器生成的 `Part` 对象列表，并将其进一步处理为指定 token 大小的块，适合在 RAG 管线中进行嵌入和检索。
    * **`Rechunker`**：
        * 根据文档标题对部分进行分组，以保持语义上下文。
        * 合并和拆分部分以符合 `chunk_size` 和 `chunk_overlap` 约束，使用提供的 `tokenizer`。
        * 将标题层级和源映射信息（来自 Markdown 或 PDF）保留并传播到最终的块中。
    * **`SimpleSemanticSplitter`**：
        * 由 `Rechunker` 用于分解过大的单个文本段。
        * 采用分层分隔符列表（段落分隔符、换行符、句子结束符等）来分割文本，同时尝试保留含义。
        * 实现块之间的重叠。

---

## 工作流程

1.  文件路径提供给 `DocParser`。
2.  `DocParser` 选择一个合适的解析器（例如，`MarkItDownParser`、`DocRayParser`）。
3.  选定的解析器处理文件：
    * 它可能将文件转换为 Markdown（例如，`MarkItDownParser`）。
    * 它可能调用外部服务进行复杂解析或 OCR/ASR（`DocRayParser`、`ImageParser`、`AudioParser`）。
4.  （可能是中间 Markdown）内容被 `parse_md.py` 解析成 `Part` 对象列表（如果适用）。此步骤将文档结构化为语义单元，如段落、标题、代码块，并处理嵌入的图像。
5.  生成的 `Part` 对象列表随后传递给 `chunking.py` 中的 `rechunk` 函数。
6.  `rechunk` 智能地组合和拆分这些部分，形成所需 token 大小的最终文本块，准备好进行嵌入。元数据，包括标题层级和源位置，与每个块相关联。

这个模块构成了 ApeRAG 中摄取管线的基础，确保各种文档类型可以被有效处理和结构化，以用于检索和生成任务。