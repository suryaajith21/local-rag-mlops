"""Layout-aware PDF parser using pdfplumber.

Extracts body text, tables (as markdown), and section headings detected by
font-size heuristics. Each block carries structured metadata for downstream
chunking and retrieval.
"""

import re
from pathlib import Path
from typing import Any

import pdfplumber


# Minimum character length to keep a block; filters page numbers, headers, footers
_MIN_BLOCK_LEN = 30

_BIBLIO_LINE_RE = re.compile(r"^\[\d+\]")


def _is_bibliography_block(text: str) -> bool:
    """Return True if the block looks like a bibliography / reference list.

    Checks whether more than 60 % of non-empty lines start with a numeric
    citation marker of the form ``[N]`` (e.g. ``[1]``, ``[42]``).  This
    pattern is reliable for academic PDFs where every reference entry begins
    with its citation number, while body text virtually never does.

    Args:
        text: The full content string of a parsed block.

    Returns:
        ``True`` if the block should be treated as a bibliography and
        skipped; ``False`` otherwise.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    matched = sum(1 for l in lines if _BIBLIO_LINE_RE.match(l))
    return matched / len(lines) > 0.60


def _rows_to_markdown(rows: list[list[Any]]) -> str:
    """Convert a list of table rows into a pipe-delimited markdown table string.

    Args:
        rows: List of rows, where each row is a list of cell values.

    Returns:
        A markdown-formatted table string, or an empty string if rows is empty.
    """
    if not rows:
        return ""

    cleaned = [[str(cell) if cell is not None else "" for cell in row] for row in rows]

    header = cleaned[0]
    separator = ["---"] * len(header)
    body = cleaned[1:] if len(cleaned) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")

    return "\n".join(lines)


def _extract_char_stats(page: pdfplumber.page.Page) -> dict[float, int]:
    """Count occurrences of each font size on the page.

    Args:
        page: A pdfplumber Page object.

    Returns:
        A mapping from font size (float) to occurrence count.
    """
    size_counts: dict[float, int] = {}
    for char in page.chars:
        size = round(float(char.get("size", 0)), 1)
        size_counts[size] = size_counts.get(size, 0) + 1
    return size_counts


def _outside_tables(
    bbox: tuple[float, float, float, float],
    table_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if bbox does not overlap any table bounding box."""
    x0, top, x1, bottom = bbox
    for tx0, ttop, tx1, tbottom in table_bboxes:
        if x0 < tx1 and x1 > tx0 and top < tbottom and bottom > ttop:
            return False
    return True


def parse_pdf(path: Path) -> list[dict]:
    """Parse a single PDF file into a list of content blocks.

    Uses pdfplumber for layout-aware extraction. Tables are formatted as
    markdown. Headings are detected when the font size is larger than the
    modal (most common) body font size on the page.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        A list of block dicts, each containing:
            - ``content`` (str): Extracted text or markdown table.
            - ``metadata`` (dict): Keys are ``source``, ``page``,
              ``block_type`` ("text" | "table" | "heading"), and ``section``
              (the most recently seen heading, or "" if none).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    blocks: list[dict] = []
    current_section: str = ""
    source_name = path.name

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # --- Step 1: Extract tables and record their bounding boxes ---
            table_bboxes: list[tuple[float, float, float, float]] = []
            for table in page.find_tables():
                rows = table.extract()
                if not rows:
                    continue
                md = _rows_to_markdown(rows)
                if len(md) >= _MIN_BLOCK_LEN:
                    blocks.append(
                        {
                            "content": md,
                            "metadata": {
                                "source": source_name,
                                "page": page_num,
                                "block_type": "table",
                                "section": current_section,
                            },
                        }
                    )
                table_bboxes.append(table.bbox)

            # --- Step 2: Extract words outside table regions ---
            try:
                words = [
                    w
                    for w in page.extract_words()
                    if _outside_tables(
                        (w["x0"], w["top"], w["x1"], w["bottom"]), table_bboxes
                    )
                ]
            except Exception:
                words = page.extract_words()

            if not words:
                continue

            # --- Step 3: Determine modal (body) font size for this page ---
            size_counts = _extract_char_stats(page)
            modal_size = max(size_counts, key=lambda s: size_counts[s]) if size_counts else 0.0

            # Build top-y → [font sizes] map for heading detection
            top_to_sizes: dict[float, list[float]] = {}
            for char in page.chars:
                ctop = round(float(char.get("top", 0)), 0)
                csize = float(char.get("size", 0))
                char_bbox = (
                    char.get("x0", 0),
                    char.get("top", 0),
                    char.get("x1", 0),
                    char.get("bottom", 0),
                )
                if _outside_tables(char_bbox, table_bboxes):
                    top_to_sizes.setdefault(ctop, []).append(csize)

            # --- Step 4: Group words into lines by y-position ---
            lines: list[tuple[float, str]] = []
            current_line_top: float | None = None
            current_line_words: list[str] = []

            for word in sorted(words, key=lambda w: (round(w["top"], 0), w["x0"])):
                top = round(float(word["top"]), 0)
                if current_line_top is None or abs(top - current_line_top) > 2:
                    if current_line_words:
                        lines.append((current_line_top, " ".join(current_line_words)))
                    current_line_top = top
                    current_line_words = [word["text"]]
                else:
                    current_line_words.append(word["text"])

            if current_line_words and current_line_top is not None:
                lines.append((current_line_top, " ".join(current_line_words)))

            # --- Step 5: Classify lines and accumulate into paragraphs ---
            paragraph_lines: list[str] = []

            def _flush_paragraph() -> None:
                nonlocal paragraph_lines
                text = " ".join(paragraph_lines).strip()
                paragraph_lines = []
                if len(text) >= _MIN_BLOCK_LEN and not _is_bibliography_block(text):
                    blocks.append(
                        {
                            "content": text,
                            "metadata": {
                                "source": source_name,
                                "page": page_num,
                                "block_type": "text",
                                "section": current_section,
                            },
                        }
                    )

            for line_top, line_text in lines:
                line_text = line_text.strip()
                if not line_text:
                    _flush_paragraph()
                    continue

                sizes_on_line = top_to_sizes.get(line_top, [])
                avg_size = (
                    sum(sizes_on_line) / len(sizes_on_line) if sizes_on_line else modal_size
                )

                # Heading heuristic: font is >10% larger than body AND line is short
                is_heading = (
                    modal_size > 0
                    and avg_size > modal_size * 1.10
                    and len(line_text) < 120
                )

                if is_heading:
                    _flush_paragraph()
                    current_section = line_text
                    blocks.append(
                        {
                            "content": line_text,
                            "metadata": {
                                "source": source_name,
                                "page": page_num,
                                "block_type": "heading",
                                "section": current_section,
                            },
                        }
                    )
                else:
                    paragraph_lines.append(line_text)

            _flush_paragraph()

    return blocks


def parse_directory(dir_path: Path) -> list[dict]:
    """Recursively parse all PDF files in a directory.

    Args:
        dir_path: Path to the directory to search.

    Returns:
        Combined list of blocks from all PDFs found, in filesystem order.
    """
    dir_path = Path(dir_path)
    all_blocks: list[dict] = []
    pdf_files = sorted(dir_path.rglob("*.pdf"))

    if not pdf_files:
        print(f"Warning: no PDF files found in {dir_path}")
        return all_blocks

    for pdf_path in pdf_files:
        print(f"Parsing: {pdf_path.name}")
        try:
            file_blocks = parse_pdf(pdf_path)
            print(f"  -> {len(file_blocks)} blocks extracted")
            all_blocks.extend(file_blocks)
        except Exception as exc:
            print(f"  Warning: failed to parse {pdf_path.name}: {exc}")

    return all_blocks
