import re
import os
import json
import logging
import argparse
import yaml
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
from openai import OpenAI
import langdetect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContentTranslator")

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


class ContentTranslator:
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

    def read_file(self, file_path: str) -> str:
        """Read the content of a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _is_yaml_file(self, file_path: str) -> bool:
        """Check if a file is a YAML file based on extension."""
        return file_path.lower().endswith(('.yaml', '.yml'))
        
    def _is_jsonl_file(self, file_path: str) -> bool:
        """Check if a file is a JSONL file based on extension."""
        return file_path.lower().endswith(('.jsonl', '.ndjson'))

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

    def split_yaml_into_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        Split YAML content into blocks, preserving structure.
        
        Args:
            content: YAML content as a string
            
        Returns:
            List of dictionaries with translatable parts and their structure
        """
        blocks = []
        
        try:
            # Parse the YAML content
            yaml_data = yaml.safe_load(content)
            
            # Process the YAML data recursively
            self._process_yaml_node(yaml_data, blocks)
            
            return blocks
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML content: {str(e)}")
            # If parsing fails, treat as a single block
            blocks.append({
                "type": "yaml_content",
                "content": content,
                "structure": None
            })
            return blocks
    
    def split_jsonl_into_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        Split JSONL content into blocks for translation
        
        Args:
            content: JSONL content as a string
            
        Returns:
            List of dictionaries with translatable parts and their structure
        """
        blocks = []
        
        try:
            # Process each line as a separate JSON object
            for i, line in enumerate(content.strip().split('\n')):
                if not line.strip():
                    continue
                    
                try:
                    # Parse the JSON object
                    json_obj = json.loads(line)
                    
                    # Process each JSON object as a separate block with its line number
                    line_blocks = []
                    self._process_json_node(json_obj, line_blocks, [])
                    
                    # Add line number information to each block
                    for block in line_blocks:
                        block["line_number"] = i
                        blocks.append(block)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON at line {i+1}: {str(e)}")
                    # Add the whole line as a single non-translatable block
                    blocks.append({
                        "type": "json_error",
                        "content": line,
                        "line_number": i
                    })
            
            return blocks
            
        except Exception as e:
            logger.error(f"Error processing JSONL file: {str(e)}")
            # If parsing fails completely, return the whole content as a block
            blocks.append({
                "type": "jsonl_content",
                "content": content,
                "structure": None
            })
            return blocks
    
    def _process_json_node(self, node: Any, blocks: List[Dict[str, Any]], path: List[str] = None) -> None:
        """
        Process a JSON node recursively, extracting translatable strings.
        
        Args:
            node: The current JSON node
            blocks: List to store translatable blocks
            path: Current path in the JSON structure
        """
        if path is None:
            path = []
        
        if isinstance(node, dict):
            # Process dictionary
            for key, value in node.items():
                new_path = path + [key]
                if isinstance(value, str) and value.strip():
                    # Translatable leaf node (only non-empty strings)
                    blocks.append({
                        "type": "json_value",
                        "content": value,
                        "path": new_path,
                        "key": key
                    })
                elif isinstance(value, (dict, list)):
                    # Recursively process nested structures
                    self._process_json_node(value, blocks, new_path)
        
        elif isinstance(node, list):
            # Process list
            for i, item in enumerate(node):
                new_path = path + [str(i)]
                if isinstance(item, str) and item.strip():
                    # Translatable leaf node (only non-empty strings)
                    blocks.append({
                        "type": "json_value",
                        "content": item,
                        "path": new_path,
                        "index": i
                    })
                elif isinstance(item, (dict, list)):
                    # Recursively process nested structures
                    self._process_json_node(item, blocks, new_path)
    
    def _process_yaml_node(self, node: Any, blocks: List[Dict[str, Any]], path: List[str] = None) -> None:
        """
        Process a YAML node recursively, extracting translatable strings.
        
        Args:
            node: The current YAML node
            blocks: List to store translatable blocks
            path: Current path in the YAML structure
        """
        if path is None:
            path = []
        
        if isinstance(node, dict):
            # Process dictionary
            for key, value in node.items():
                new_path = path + [key]
                if isinstance(value, (str, int, float, bool)) and isinstance(value, str):
                    # Translatable leaf node
                    blocks.append({
                        "type": "yaml_value",
                        "content": value,
                        "path": new_path,
                        "key": key
                    })
                elif isinstance(value, (dict, list)):
                    # Recursively process nested structures
                    self._process_yaml_node(value, blocks, new_path)
        
        elif isinstance(node, list):
            # Process list
            for i, item in enumerate(node):
                new_path = path + [str(i)]
                if isinstance(item, (str, int, float, bool)) and isinstance(item, str):
                    # Translatable leaf node
                    blocks.append({
                        "type": "yaml_value",
                        "content": item,
                        "path": new_path,
                        "index": i
                    })
                elif isinstance(item, (dict, list)):
                    # Recursively process nested structures
                    self._process_yaml_node(item, blocks, new_path)

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

    def translate_jsonl_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a JSONL value block.
        
        Args:
            block: Dictionary with JSON value and structure info
            
        Returns:
            Dictionary with translated content and original structure info
        """
        # Skip non-translatable blocks
        if block["type"] not in ["json_value"]:
            return block
            
        content = block["content"]
        
        # Don't translate if it's not a string or empty
        if not isinstance(content, str) or not content.strip():
            return block
        
        # Get the language (either specified or detected)
        detected_lang = self.detect_language(content)
        
        # If it's already in the target language or can't be detected, return original
        if detected_lang == self.target_lang or (detected_lang == 'unknown' and not self.source_lang):
            return block
        
        # Get language names for clearer prompts
        source_lang_name = self._get_language_name(
            detected_lang) if detected_lang != 'unknown' else "the source language"
        target_lang_name = self._get_language_name(self.target_lang)
        
        # Prepare translation prompt
        system_prompt = f"""You are a JSON content translator.
        
        Rules:
        1. Translate the text from {source_lang_name} to {target_lang_name}
        2. Preserve any special formatting, variables, or placeholders exactly as they appear
        3. Return ONLY the translated text without any additional text
        4. Keep line breaks exactly as they appear in the original
    5. DO NOT include tags like "<TRANSLATE>" or "</TRANSLATE>" in your response
    6. Respond ONLY with the translated text and nothing else

    For this {block["type"]}:
    {special_instructions.get(block["type"], "")} in the original
        5. Maintain the exact same meaning in the translation
        6. Do not add any quotes, brackets, braces, or other JSON syntax - just translate the text content
        """
        
        user_prompt = f"""Translate this JSON value from {source_lang_name} to {target_lang_name}, preserving all formatting and variables:

        <TRANSLATE>
        {content}
        </TRANSLATE>

        IMPORTANT: Return only the translated text. Do not include any tags, explanations, or other markers in your response.
        Preserve all variable placeholders, brackets, braces like {{variable}}, and special characters exactly as they appear.
        """
        
        try:
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
            translated_text = self._clean_response(translated_text)
            
            # Return updated block
            translated_block = block.copy()
            translated_block["content"] = translated_text
            
            return translated_block
            
        except Exception as e:
            logger.error(f"Error translating JSON block: {str(e)}")
            return block  # Return original block if translation fails

    def translate_yaml_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a YAML value block.
        
        Args:
            block: Dictionary with YAML value and structure info
            
        Returns:
            Dictionary with translated content and original structure info
        """
        content = block["content"]
        
        # Don't translate if it's not a string
        if not isinstance(content, str) or not content.strip():
            return block
        
        # Get the language (either specified or detected)
        detected_lang = self.detect_language(content)
        
        # If it's already in the target language or can't be detected, return original
        if detected_lang == self.target_lang or (detected_lang == 'unknown' and not self.source_lang):
            return block
        
        # Get language names for clearer prompts
        source_lang_name = self._get_language_name(
            detected_lang) if detected_lang != 'unknown' else "the source language"
        target_lang_name = self._get_language_name(self.target_lang)
        
        # Prepare translation prompt
        system_prompt = f"""You are a YAML content translator.
        
        Rules:
        1. Translate the text from {source_lang_name} to {target_lang_name}
        2. Preserve any special formatting, variables, or placeholders exactly as they appear
        3. Return ONLY the translated text without any additional text
        4. Keep line breaks exactly as they appear in the original
        5. Maintain the exact same meaning in the translation
        """
        
        user_prompt = f"""Translate this YAML value from {source_lang_name} to {target_lang_name}, preserving all formatting and variables:

        <TRANSLATE>
        {content}
        </TRANSLATE>

        IMPORTANT: Return only the translated text. Do not include any tags, explanations, or other markers in your response.
        Preserve all variable placeholders, brackets, braces like {{variable}}, and special characters exactly as they appear.
        """
        
        try:
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
            translated_text = self._clean_response(translated_text)
            
            # Return updated block
            translated_block = block.copy()
            translated_block["content"] = translated_text
            
            return translated_block
            
        except Exception as e:
            logger.error(f"Error translating YAML block: {str(e)}")
            return block  # Return original block if translation fails

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
    4. Keep line breaks exactly as they appear
