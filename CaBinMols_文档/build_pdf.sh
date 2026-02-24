#!/usr/bin/env bash
set -euo pipefail

# Compile main.tex into main.pdf and clean auxiliary files.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEX_FILE="main.tex"
PDF_FILE="main.pdf"

if [[ ! -f "$TEX_FILE" ]]; then
  echo "Error: $TEX_FILE not found in $SCRIPT_DIR"
  exit 1
fi

if command -v latexmk >/dev/null 2>&1; then
  echo "[1/2] Compiling with latexmk (xelatex)..."
  latexmk -xelatex -interaction=nonstopmode -file-line-error "$TEX_FILE"

  echo "[2/2] Cleaning auxiliary files..."
  latexmk -c "$TEX_FILE"
else
  echo "Warning: latexmk not found, fallback to xelatex+bibtex."
  echo "[1/2] Compiling with xelatex/bibtex..."
  xelatex -interaction=nonstopmode -file-line-error "$TEX_FILE"
  bibtex "${TEX_FILE%.tex}" || true
  xelatex -interaction=nonstopmode -file-line-error "$TEX_FILE"
  xelatex -interaction=nonstopmode -file-line-error "$TEX_FILE"

  echo "[2/2] Cleaning auxiliary files..."
  rm -f \
    "${TEX_FILE%.tex}.aux" \
    "${TEX_FILE%.tex}.bbl" \
    "${TEX_FILE%.tex}.blg" \
    "${TEX_FILE%.tex}.fdb_latexmk" \
    "${TEX_FILE%.tex}.fls" \
    "${TEX_FILE%.tex}.log" \
    "${TEX_FILE%.tex}.out" \
    "${TEX_FILE%.tex}.toc" \
    "${TEX_FILE%.tex}.synctex.gz" \
    "${TEX_FILE%.tex}.lof" \
    "${TEX_FILE%.tex}.lot" \
    "${TEX_FILE%.tex}.bcf" \
    "${TEX_FILE%.tex}.run.xml"
fi

if [[ -f "$PDF_FILE" ]]; then
  echo "Done: generated $PDF_FILE"
else
  echo "Error: failed to generate $PDF_FILE"
  exit 1
fi
