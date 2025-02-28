# MarkPoly

`MarkPoly` is a smart polyglot Markdown translation that preserves meaning in text and code.
- Handles Markdown documents
- Supports multiple languages
- Preserves formatting while translating
- Intelligently handles code blocks and comments

In short, an LLM powered Markdown translator that understands both prose and code.

**Features:**
- Code Block Translation:
  - Handles Markdown paragraphs, lists, tables and code blocks
  - Handles code blocks separately from other markdown content
  - Extracts and identifies code comments
  - Preserves code structure and only translates comments
  - Maintains comment markers and indentation
  - Supports both single-line and multi-line comments

- Regular Markdown Translation:
  - Different handling for tables, lists, headers, and paragraphs
  - Preserves markdown formatting for each type
  - Maintains special characters and structure
  - Translate between two languages

- Language Detection and Handling:
  - Uses specified source language if provided
  - Falls back to automatic detection
  - Skips translation for English content
  - Handles unknown language cases
  - Fallback to original content on error
  - Preservation of markdown formatting

## Usage

As prerequisite a polyglot LLM like `granite3.1-dense:latest` should be available over an OpenAI compatible API endpoint (e.g. Ollama, vLLM, llama_cpp, etc.)

```bash
# example translating markdown file from Spanish to English
python markpoly.py --source-lang es ejemplo.md

# after running it will create a translated file `ejemplo_translated.md`
```

Additional arguments and their usage:

```bash
markpoly.py [-h] [-o OUTPUT] [--source-lang SOURCE_LANG] [--api-key API_KEY] [--base-url BASE_URL] [--model MODEL] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--quiet] input_file


Translate markdown files between languages using AI.

positional arguments:
  input_file            Path to the input markdown file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to save the translated markdown file. If not provided, will use input_file_translated.md (default: None)
  --source-lang SOURCE_LANG
                        Source language code or name (e.g., "es" or "spanish") (default: None)
  --target-lang TARGET_LANG
                        Target language code or name (e.g., "fr" or "french"), defaults to English (default: en)
  --api-key API_KEY     API key for the translation service (default: no-key)
  --base-url BASE_URL   Base URL for the API endpoint (default: http://localhost:11434/v1/)
  --model MODEL         Model name to use for translation (default: granite3.1-dense:latest)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: INFO)
  --quiet               Suppress progress information (default: False)
  --list-languages      List all supported languages and exit (default: False)
```
## Known Issues
- It has been developed and tested with `granite-3.1`.
- It can work with `Phi4`, `qwen2`, and other multilingual models that support `system` and `user` roles but performance varies.
- It ***DOES NOT*** work with `Mistral`/`Mixtral` family of models as they do not use a `system` prompt role and the logic for combining the `system` and `user` prompts needs to be coded.