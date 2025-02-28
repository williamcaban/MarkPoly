import re
import os
import json
import logging
import argparse
from typing import List, Dict, Optional
from pathlib import Path
from openai import OpenAI
import langdetect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarkdownTranslator")

# Dictionary of supported languages and their codes
SUPPORTED_LANGUAGES = {
    'arabic': 'ar',     # granite-3.1-8b-instruct
    'chinese': 'zh',    # granite-3.1-8b-instruct
    'czech': 'cs',      # granite-3.1-8b-instruct
    'dutch': 'nl',      # granite-3.1-8b-instruct
    'english': 'en',    # granite-3.1-8b-instruct
    'french': 'fr',     # granite-3.1-8b-instruct
    'german': 'de',     # granite-3.1-8b-instruct
    'hindi': 'hi',
    'italian': 'it',    # granite-3.1-8b-instruct
    'japanese': 'ja',   # granite-3.1-8b-instruct
    'korean': 'ko',     # granite-3.1-8b-instruct
    'portuguese': 'pt',  # granite-3.1-8b-instruct
    'russian': 'ru',
    'spanish': 'es',    # granite-3.1-8b-instruct
    'swedish': 'sv',
    'turkish': 'tr',
}


class MarkdownTranslator:
    def __init__(
        self,
        api_key: str = "no-key",
        base_url: str = "http://localhost:11434/v1/",
        model: str = "granite3.1-dense:latest",
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = 'en'
    ):
        """
        Initialize the translator with API credentials and endpoint configuration.
        
        Args:
            api_key: API key for the translation service
            base_url: Base URL for the API endpoint
            model: Model name to use for translation
            source_lang: Optional source language code or name (e.g., 'es' or 'spanish')
            target_lang: Target language code or name (e.g., 'fr' or 'french'), defaults to English
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.source_lang = self._normalize_language_code(
            source_lang) if source_lang else None
        self.target_lang = self._normalize_language_code(
            target_lang) if target_lang else 'en'
        logger.info(f"Initialized translator with model: {model}")
        logger.info(f"Using API endpoint: {base_url}")
        if self.source_lang:
            logger.info(
                f"Source language explicitly set to: {self.source_lang}")
        logger.info(f"Target language set to: {self.target_lang}")

    def _normalize_language_code(self, lang: str) -> str:
        """
        Normalize language input to standard language code.
        
        Args:
            lang: Language name or code (e.g., 'japanese' or 'ja')
            
        Returns:
            Normalized language code
        """
        if not lang:
            return None

        lang = lang.lower()

        # If it's already a valid language code, return it
        if lang in SUPPORTED_LANGUAGES.values():
            return lang

        # If it's a language name, convert to code
        if lang in SUPPORTED_LANGUAGES:
            return SUPPORTED_LANGUAGES[lang]

        # If we can't normalize it, log a warning and return as-is
        logger.warning(f"Unrecognized language specification: {lang}")
        return lang

    def _get_language_name(self, lang_code: str) -> str:
        """
        Get the language name from a language code.
        
        Args:
            lang_code: Language code (e.g., 'ja')
            
        Returns:
            Language name (e.g., 'Japanese')
        """
        for name, code in SUPPORTED_LANGUAGES.items():
            if code == lang_code:
                return name.capitalize()
        return lang_code.upper()

    def read_markdown_file(self, file_path: str) -> str:
        """Read the content of a markdown file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_into_blocks(self, content: str) -> List[Dict[str, str]]:
        """
        Split markdown content into individual blocks (paragraphs, headers, tables, or lists).
        
        Args:
            content: The markdown content as a string
            
        Returns:
            List of dictionaries containing block type and content
        """
        blocks = []
        current_lines = []
        current_type = "paragraph"

        def is_table_line(line: str) -> bool:
            """Check if a line is part of a markdown table."""
            stripped = line.strip()
            return stripped.startswith('|') or stripped.startswith('+-') or stripped.startswith('|-')

        def is_list_line(line: str) -> bool:
            """Check if a line is part of a markdown list."""
            stripped = line.strip()
            # Match numbered lists (1., 2., etc.) or bullet lists (-, *, +)
            return bool(re.match(r'^\d+\.|\s*[-*+]', stripped))

        def is_indented(line: str) -> bool:
            """Check if a line is indented (part of a list or code block)."""
            return line.startswith('    ') or line.startswith('\t')

        def add_block(lines, type_="paragraph"):
            if lines:
                content = '\n'.join(lines).strip()
                if content:  # Only add non-empty blocks
                    blocks.append({
                        "type": type_,
                        "content": content
                    })

        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Handle headers
            if stripped.startswith('#'):
                add_block(current_lines, current_type)
                current_lines = [line]
                add_block(current_lines, "header")
                current_lines = []
                current_type = "paragraph"

            # Handle tables
            elif is_table_line(line):
                if current_type != "table":
                    add_block(current_lines, current_type)
                    current_lines = []
                    current_type = "table"
                current_lines.append(line)

            # Handle lists
            elif is_list_line(line) or (current_type == "list" and (stripped or is_indented(line))):
                if current_type != "list":
                    add_block(current_lines, current_type)
                    current_lines = []
                    current_type = "list"
                current_lines.append(line)

            # Handle code blocks
            elif stripped.startswith('```'):
                if current_type != "code":
                    add_block(current_lines, current_type)
                    current_lines = []
                    current_type = "code"
                current_lines.append(line)
                # Skip until we find the closing code block
                if len(current_lines) == 1:  # Opening fence
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith('```'):
                        current_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        current_lines.append(lines[i])  # Add closing fence

            # Handle empty lines
            elif not stripped:
                if current_type in ["table", "list", "code"]:
                    # For tables, lists and code blocks, preserve empty lines
                    current_lines.append(line)
                else:
                    # For paragraphs, use empty lines as separators
                    add_block(current_lines, current_type)
                    current_lines = []
                    current_type = "paragraph"

            # Handle regular paragraph lines
            else:
                if current_type in ["table", "list", "code"]:
                    # Check if we're actually continuing the special block
                    if (current_type == "table" and not is_table_line(line)) or \
                       (current_type == "list" and not (is_list_line(line) or is_indented(line))):
                        add_block(current_lines, current_type)
                        current_lines = []
                        current_type = "paragraph"
                current_lines.append(line)

            i += 1

        # Add the final block
        add_block(current_lines, current_type)

        return blocks

    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text while ignoring markdown formatting.
        Returns explicitly set language if specified, otherwise attempts detection.
        
        Args:
            text: Text to detect language from
            
        Returns:
            Language code (e.g., 'en' for English)
        """
        # If source language is explicitly set, use it
        if self.source_lang:
            return self.source_lang

        try:
            # Remove markdown formatting for better language detection
            cleaned_text = re.sub(r'[#*_`|[-\]()]+', ' ', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

            if not cleaned_text:  # If text is empty or only contains markdown
                return 'unknown'

            return langdetect.detect(cleaned_text)
        except langdetect.LangDetectException:
            return 'unknown'

    def _extract_code_comments(self, code: str) -> List[Dict[str, str]]:
        """
        Extract comments from code while preserving their position and format.
        
        Args:
            code: String containing code with comments
            
        Returns:
            List of dictionaries containing comment info (text, start_pos, end_pos, type)
        """
        comments = []
        lines = code.split('\n')
        current_pos = 0

        def is_in_string(line: str, pos: int) -> bool:
            """Check if position is inside a string literal."""
            in_single = False
            in_double = False
            escaped = False

            for i in range(pos):
                if line[i] == '\\':
                    escaped = not escaped
                    continue

                if not escaped:
                    if line[i] == "'":
                        if not in_double:
                            in_single = not in_single
                    elif line[i] == '"':
                        if not in_single:
                            in_double = not in_double

                escaped = False

            return in_single or in_double

        i = 0
        while i < len(lines):
            line = lines[i]
            line_start = current_pos

            # Handle single-line comments
            if '//' in line:  # JavaScript/C-style
                pos = line.index('//')
                if not is_in_string(line, pos):
                    comments.append({
                        'text': line[pos:],
                        'start_pos': line_start + pos,
                        'end_pos': line_start + len(line),
                        'type': 'single'
                    })

            elif '#' in line:  # Python-style
                pos = line.index('#')
                if not is_in_string(line, pos):
                    comments.append({
                        'text': line[pos:],
                        'start_pos': line_start + pos,
                        'end_pos': line_start + len(line),
                        'type': 'single'
                    })

            # Handle multi-line comments
            elif '/*' in line:  # JavaScript/C-style
                pos = line.index('/*')
                if not is_in_string(line, pos):
                    comment_text = [line[pos:]]
                    comment_start = line_start + pos
                    i += 1

                    # Collect comment lines until we find the closing */
                    while i < len(lines):
                        if '*/' in lines[i]:
                            end_pos = lines[i].index('*/') + 2
                            comment_text.append(lines[i][:end_pos])
                            comments.append({
                                'text': '\n'.join(comment_text),
                                'start_pos': comment_start,
                                'end_pos': current_pos + len(lines[i]),
                                'type': 'multi'
                            })
                            break
                        else:
                            comment_text.append(lines[i])
                            i += 1
                            current_pos += len(lines[i]) + 1

            elif '"""' in line or "'''" in line:  # Python docstring
                if line.count('"""') == 1 or line.count("'''") == 1:  # Opening quote
                    marker = '"""' if '"""' in line else "'''"
                    pos = line.index(marker)
                    if not is_in_string(line[:pos], pos):
                        comment_text = [line[pos:]]
                        comment_start = line_start + pos
                        i += 1

                        # Collect docstring lines until we find the closing quotes
                        while i < len(lines):
                            if marker in lines[i]:
                                end_pos = lines[i].index(marker) + 3
                                comment_text.append(lines[i][:end_pos])
                                comments.append({
                                    'text': '\n'.join(comment_text),
                                    'start_pos': comment_start,
                                    'end_pos': current_pos + end_pos,
                                    'type': 'docstring'
                                })
                                break
                            else:
                                comment_text.append(lines[i])
                                i += 1
                                current_pos += len(lines[i]) + 1

            current_pos += len(line) + 1
            i += 1

        return comments

    def translate_block(self, block: Dict[str, str]) -> Dict[str, str]:
        """
        Translate a single block using the configured API.
        
        Args:
            block: Dictionary containing block type and content
            
        Returns:
            Dictionary with translated content
        """
        # For code blocks, only translate comments
        if block["type"] == "code":
            # Extract the actual code content (remove the ```language and ``` markers)
            code_lines = block["content"].split('\n')
            if len(code_lines) >= 2:  # Valid code block with markers
                code_content = '\n'.join(code_lines[1:-1])  # Remove markers
                code_language = code_lines[0].replace('```', '').strip()

                # Extract comments
                comments = self._extract_code_comments(code_content)
                if not comments:  # No comments to translate
                    return block

                # Translate each comment
                translated_code = code_content
                # Process from end to avoid position shifts
                for comment in reversed(comments):
                    comment_text = comment['text']
                    detected_lang = self.detect_language(comment_text)

                    # Only translate if source and target languages are different
                    if detected_lang != self.target_lang and detected_lang != 'unknown':
                        # Prepare translation prompt for comment
                        target_lang_name = self._get_language_name(
                            self.target_lang)
                        system_prompt = f"""You are a code comment translator.
                        Translate only the comment text from {self._get_language_name(detected_lang)} to {target_lang_name} while preserving any comment markers and formatting.
                        If the comment contains code examples, variable names, or function names, keep those unchanged.
                        Preserve any special comment markers (e.g., @param, @return, TODO:, FIXME:, etc.)."""

                        user_prompt = f"Translate this code comment to {target_lang_name}:\n\n{comment_text}"

                        try:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.3
                            )

                            translated_comment = response.choices[0].message.content.strip(
                            )
                            # Replace the comment in the code while preserving indentation
                            translated_code = (
                                translated_code[:comment['start_pos']] +
                                translated_comment +
                                translated_code[comment['end_pos']:]
                            )
                        except Exception as e:
                            logger.error(
                                f"Error translating comment: {str(e)}")

                # Reconstruct the code block with markers
                return {
                    "type": "code",
                    "content": f"```{code_language}\n{translated_code}\n```"
                }

        # For non-code blocks, handle regular markdown translation
        special_instructions = {
            "table": "- Output the table with exact same structure, only translated content in cells\n- Keep all | and - characters in their exact positions",
            "list": "- Output the list with exact same markers and indentation\n- Keep all list markers (-, *, +, numbers) exactly as they appear",
            "code": "- Output the code block with exact same structure\n- Only translate comments, keep code unchanged",
            "header": "- Output the header with exact same # symbols\n- Keep same number of # characters at the start",
            "paragraph": "- Output the paragraph with exact same inline formatting\n- Keep all *bold*, _italic_, [links](urls) exactly as they appear"
        }

        # Get the language (either specified or detected)
        detected_lang = self.detect_language(block["content"])
        logger.debug(
            f"Language: {detected_lang} ({'specified' if self.source_lang else 'detected'})")

        # If it's already in the target language or can't be detected, return original
        if detected_lang == self.target_lang or (detected_lang == 'unknown' and not self.source_lang):
            logger.debug(
                f"Text is already in {self._get_language_name(self.target_lang)} or language cannot be detected. Skipping translation.")
            return block

        # Get language names for clearer prompts
        source_lang_name = self._get_language_name(
            detected_lang) if detected_lang != 'unknown' else "the source language"
        target_lang_name = self._get_language_name(self.target_lang)

        # Updated system prompt to be more explicit about not including the markers
        system_prompt = f"""You are a markdown translator that only outputs the translated content.

    Rules:
    1. Translate the content between the tags "<TRANSLATE>" and "</TRANSLATE>" from {source_lang_name} to {target_lang_name}
    2. Keep all markdown formatting exactly as is
    3. Return ONLY the translated content without ANY additional text
    4. Keep line breaks exactly as they appear in the original
    5. DO NOT include tags like "<TRANSLATE>" or "</TRANSLATE>" in your response
    6. Respond ONLY with the translated text and nothing else

    For this {block["type"]}:
    {special_instructions.get(block["type"], "")}"""

        user_prompt = f"""Translate this {block["type"]} from {source_lang_name} to {target_lang_name}, preserving all formatting.

    <TRANSLATE>
    {block['content']}
    </TRANSLATE>

    IMPORTANT: Output only the translated text. Do not include any tags, or other markers in your response. Preserve the exact same markdown formatting of the original content."""

        try:
            # Log the request
            logger.debug("\n=== REQUEST ===")
            logger.debug(f"Model: {self.model}")
            logger.debug("Messages:")
            logger.debug(json.dumps([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], ensure_ascii=False, indent=2))

            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            translated_text = response.choices[0].message.content.strip()

            # Clean up any unwanted elements from the response
            ##translated_text = self._clean_response(translated_text)

            # Log the response
            logger.debug("\n=== RESPONSE ===")
            logger.debug(f"Response content:")
            logger.debug(json.dumps(translated_text,
                        ensure_ascii=False, indent=2))

            # Verify the translation preserves markdown structure
            if block["type"] == "header" and not translated_text.startswith('#'):
                logger.warning(
                    "Translation lost header formatting, attempting to restore")
                original_hashes = re.match(r'^#+', block["content"]).group(0)
                translated_text = f"{original_hashes} {translated_text.lstrip('#').lstrip()}"

            return {
                "type": block["type"],
                "content": translated_text
            }

        except Exception as e:
            logger.error(f"Error translating block: {str(e)}")
            return block  # Return original block if translation fails


    def _clean_response(self, text: str) -> str:
        """
        Clean up any unwanted tags or markers from the translated text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text with unwanted elements removed
        """
        # Remove triple backticks at the beginning and end
        text = re.sub(r'^```[a-zA-Z]*\s*', '', text)
        text = re.sub(r'\s *```\s*', '', text)

        # Remove any instruction repetition that might be included
        patterns_to_remove = [
            r'INSTRUCTIONS:.*$',
            r'IMPORTANT:.*$',
            r'Output only the translated text\..*$',
            r'Do not include.*$',
            r'Translated content:.*$',
            r'Translation:.*$'
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)

        return text.strip()

    def translate_markdown(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        verbose: bool = True,
        log_level: str = "INFO"
    ) -> str:
        """
        Translate an entire markdown file and save the result.
        
        Args:
            input_file: Path to input markdown file
            output_file: Optional path to save translated file
            source_lang: Optional source language override
            target_lang: Optional target language override
            verbose: Whether to print progress information
            log_level: Logging level to use
            
        Returns:
            Translated markdown content as string
        """
        # Set logging level for this translation
        logger.setLevel(getattr(logging, log_level.upper()))

        # Save original language settings
        original_source_lang = self.source_lang
        original_target_lang = self.target_lang

        # Allow overriding source and target languages for this specific translation
        if source_lang:
            self.source_lang = self._normalize_language_code(source_lang)
            logger.info(
                f"Source language temporarily set to: {self.source_lang}")

        if target_lang:
            self.target_lang = self._normalize_language_code(target_lang)
            logger.info(
                f"Target language temporarily set to: {self.target_lang}")

        try:
            content = self.read_markdown_file(input_file)
            blocks = self.split_into_blocks(content)

            translated_blocks = []
            total_blocks = len(blocks)

            for i, block in enumerate(blocks, 1):
                if verbose:
                    block_preview = block["content"][:50] + \
                        "..." if len(block["content"]
                                     ) > 50 else block["content"]
                    logger.info(
                        f"Processing block {i}/{total_blocks} ({block['type']}): {block_preview}")

                translated = self.translate_block(block)
                translated_blocks.append(translated)

            # Combine translated blocks with appropriate spacing
            translated_content = ""
            for i, block in enumerate(translated_blocks):
                if i > 0:  # Add spacing between blocks
                    if block["type"] == "header" or translated_blocks[i-1]["type"] == "header":
                        translated_content += "\n\n"
                    elif block["type"] == "paragraph" and translated_blocks[i-1]["type"] == "paragraph":
                        translated_content += "\n\n"
                translated_content += block["content"]

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                if verbose:
                    logger.info(f"Translation saved to: {output_file}")

            return translated_content

        finally:
            # Restore original language settings
            self.source_lang = original_source_lang
            self.target_lang = original_target_lang


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Translate markdown files between languages using AI.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input markdown file'
    )

    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to save the translated markdown file. If not provided, will use input_file_translated.md'
    )
    parser.add_argument(
        '--source-lang',
        type=str,
        help='Source language code or name (e.g., "es" or "spanish")'
    )
    parser.add_argument(
        '--target-lang',
        type=str,
        default='en',
        help='Target language code or name (e.g., "fr" or "french"), defaults to English'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv("API_KEY", "no-key"),
        help='API key for the translation service'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=os.getenv("BASE_URL", "http://localhost:11434/v1/"),
        help='Base URL for the API endpoint'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=os.getenv("MODEL_NAME", "granite3.1-dense:latest"),
        help='Model name to use for translation'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=os.getenv("LOG_LEVEL", "INFO"),
        help='Set the logging level'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress information'
    )
    parser.add_argument(
        '--list-languages',
        action='store_true',
        help='List all supported languages and exit'
    )

    args = parser.parse_args()

    # If the user just wants to list supported languages
    if args.list_languages:
        print("Supported Languages:")
        for name, code in sorted(SUPPORTED_LANGUAGES.items()):
            print(f"  {name.capitalize()} ({code})")
        return 0

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("MarkdownTranslator")

    # Log configuration
    logger.info(f"Using model: {args.model}")
    logger.info(f"Using base URL: {args.base_url}")

    # Verify input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1

    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.input_file)
        target_lang_code = args.target_lang.lower()
        target_lang_code = next((code for name, code in SUPPORTED_LANGUAGES.items()
                                 if name.lower() == target_lang_code.lower()
                                 or code.lower() == target_lang_code.lower()),
                                target_lang_code)
        args.output = str(
            input_path.parent / f"{input_path.stem}_{target_lang_code}{input_path.suffix}")

    # Create translator instance
    translator = MarkdownTranslator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        source_lang=args.source_lang,
        target_lang=args.target_lang
    )

    try:
        translated_content = translator.translate_markdown(
            args.input_file,
            args.output,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            verbose=not args.quiet,
            log_level=args.log_level
        )
        logger.info(
            f"Translation completed successfully! Output saved to: {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
