"""Single-shot assembler for results/FULL_PAPER_FACT_CHECK.md.

Reads the chunk .txt files in results/_fpc_chunks/ in sorted order, concatenates them with a
single blank-line separator, and writes the full markdown to results/FULL_PAPER_FACT_CHECK.md
in ONE Path.write_text() call. This is the single file-write operation.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
chunks_dir = ROOT / 'results' / '_fpc_chunks'
out_path = ROOT / 'results' / 'FULL_PAPER_FACT_CHECK.md'

# Each chunk begins with heading text and does NOT end with a trailing newline,
# so we join with "\n\n" to create a single blank line between sections.
parts = []
for chunk in sorted(chunks_dir.glob('*.txt')):
    parts.append(chunk.read_text(encoding='utf-8').rstrip('\n'))

REPORT = '\n\n'.join(parts) + '\n'
out_path.write_text(REPORT, encoding='utf-8')
print('WROTE', out_path)
print('bytes:', len(REPORT.encode('utf-8')))
print('lines:', REPORT.count(chr(10)))
