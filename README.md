[![CI](https://github.com/juanjosecas/markovify/actions/workflows/ci.yml/badge.svg)](https://github.com/juanjosecas/markovify/actions/workflows/ci.yml)

# Markovify

Extensible discrete Markov chains, text generation, document ingestion and NLP corpus preprocessing in Python. This repository extends the original [Markovify](https://github.com/jsvine/markovify) by Jeremy Singer-Vine while retaining its core API.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Document ingestion](#document-ingestion)
- [NLP preprocessing](#nlp-preprocessing)
- [End-to-end workflow](#end-to-end-workflow)
- [Constrained text generation](#constrained-text-generation)
- [Model persistence](#model-persistence)
- [General Markov chains](#general-markov-chains)
- [Combining and compiling models](#combining-and-compiling-models)
- [Customization and development](#customization-and-development)
- [Current limitations](#current-limitations)

## Installation

Python >= 3.9 is declared by the package. Install from **this repository**, rather than the original PyPI distribution, to obtain the extensions described here. Python 3.11 is a practical environment choice for optional dependencies.

### Direct installation from GitHub

Requires Git and Python.

```bash
python -m pip install "markovify @ git+https://github.com/juanjosecas/markovify.git@master"
```

With all optional features, including development and testing dependencies:

```bash
python -m pip install "markovify[all] @ git+https://github.com/juanjosecas/markovify.git@master"
```

For reproducibility, replace master with a specific commit SHA.

### Local installation (editable development)

```bash
git clone https://github.com/juanjosecas/markovify.git
cd markovify
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The editable flag (`-e`) imports the package from the checkout, so changes to Python source files are available without reinstalling (restart running Python sessions as needed). Reinstall after changing dependency or packaging metadata.

For a regular, non-editable local installation use `python -m pip install .`.

Alternatively, with Conda or Miniforge:

```bash
conda create -n markovify python=3.11 pip
conda activate markovify
git clone https://github.com/juanjosecas/markovify.git
cd markovify
python -m pip install -e ".[all]"
```

| Extra | Dependencies / purpose |
| --- | --- |
| none | Core Markov chains and basic corpus utilities; Unidecode |
| nlp | spaCy, NLTK |
| documents | pypdf, python-docx, odfpy, striprtf, ebooklib |
| hdf5 | h5py |
| dev | pytest, pytest-cov, build, h5py |
| notebooks | JupyterLab, pandas, matplotlib |
| all | All optional runtime, development and notebook dependencies |

Verify the imported checkout:

```bash
python -c "import markovify; print(markovify.__version__); print(markovify.__file__)"
```

Update an editable checkout with `git pull origin master`, then repeat the editable installation if dependencies changed.

## Quick start

```python
import markovify

model = markovify.NewlineText(
    "The protein binds the ligand.\nThe protein binds the receptor.",
    state_size=2,
    random_state=42,
)
print(model.make_sentence(test_output=False))
print(model.make_short_sentence(280, test_output=False))
```

Use Text for prose and NewlineText for one sentence per line. State size controls the number of preceding tokens used to predict the next token. Text defaults to 2. The default novelty filter rejects excessive overlap with retained training text; generation returns None when attempts are exhausted. Small demonstration corpora often require test_output=False, which permits reproducing training sentences.

## Document ingestion

```python
from markovify.nlp import Corpus, supported_formats, discover_files, read_document

print(supported_formats())
single = Corpus.load("paper.pdf")
mixed = Corpus.load(["paper.pdf", "notes.docx", "./books"])
all_documents = Corpus.load("./library")
selected = Corpus.load("./library", formats=["pdf", ".docx", "TXT"],
                       recursive=True, errors="skip")
```

Directory searches are recursive by default, sorted within each directory, and deduplicate resolved paths. Unsupported extensions encountered in directories are ignored. Explicit unsupported paths or missing paths raise errors. Format filters accept extensions with or without a leading dot, case-insensitively.

| Extensions | Extraction |
| --- | --- |
| txt, log, md, markdown | One entry per line; Markdown syntax retained |
| html, htm | HTML text |
| pdf | One entry per page, using pypdf |
| docx | Paragraphs followed by table cells |
| odt | Paragraphs |
| rtf | Extracted text |
| epub | Document items |
| csv, tsv | Selected column |
| json, jsonl | Strings or selected record field |

```python
abstracts = Corpus.load("./datasets", formats=["csv", "jsonl"],
                        text_field="abstract")
pages = read_document("paper.pdf")
paths = list(discover_files("./library", formats=["pdf"], recursive=False))
```

Use encoding="utf-8" (default) or another appropriate encoding for text readers. The default structured-data field is "text".

### Extraction audit

```python
import json

raw = Corpus.load("./library", errors="skip")
for row in raw.load_report:
    print(row["path"], row["status"], row["entries"])

with open("extraction_report.json", "w", encoding="utf-8") as f:
    json.dump(raw.load_report, f, ensure_ascii=False, indent=2)

for source, entry in zip(raw.sources, raw.texts):
    print(source, entry[:100])
```

The default errors="raise" propagates extraction failures; errors="skip" records per-file failures and continues. Discovery errors are not suppressed. Status is ok, empty or error. Inspect the report before training. Source tracking is available on the loaded corpus, not automatically propagated through transformations.

## NLP preprocessing

### Cleaning and corpus operations

```python
from markovify.nlp import Corpus, clean_text

raw = Corpus(["<p>Hello world.</p>", "Hello world.", ""])
clean = (raw.map(clean_text, html=True, urls=True, emails=True)
            .filter_length(min_words=2)
            .deduplicate(case_sensitive=False))
clean.save("clean.txt")
clean.save("clean.jsonl")
```

Corpus transformations return new objects. Available operations:

| Method | Purpose |
| --- | --- |
| map(function, **kwargs) | Transform each entry |
| split(function, **kwargs) | Expand each entry into multiple entries |
| filter(predicate) | Retain matching entries |
| filter_length(min_words, max_words) | Whitespace-based word limits |
| deduplicate(case_sensitive=True) | Exact duplicate removal |
| to_text(separator="\\n") | Join entries |
| to_model(tokenizer=None, **kwargs) | Train NewlineText on presegmented entries |
| save(path, encoding="utf-8", text_field="text") | TXT or JSONL output |

TXT requires single-line entries; use JSONL for multiline text. Saving overwrites the target file.

Standalone cleaning functions: strip_html, normalize_unicode, normalize_whitespace, remove_controls, remove_urls, remove_emails, remove_accents, lowercase. clean_text supports html, urls, emails, accents, lower and unicode_form options; destructive transformations are opt-in.

### spaCy

```bash
python -m spacy download es_core_news_sm
```

```python
from markovify.nlp import SpacyNLP

nlp = SpacyNLP(language="es")  # blank tokenizer + sentence segmenter
annotator = SpacyNLP(model="es_core_news_sm")
text = "Las proteínas interactúan con los ligandos."

print(nlp.tokenize(text))
print(nlp.sentences(text))
print(annotator.pos(text))
print(annotator.entities(text))
print(annotator.transform(text, lemmatize=True, remove_stopwords=True,
                          remove_punctuation=True, keep_pos=["NOUN", "VERB"]))
```

The wrapper exposes the original spaCy Language object as nlp.nlp, Doc via doc(text), and batched processing via pipe(texts, **kwargs). transform also accepts remove_numbers and min_length. Lemmatization, POS and NER require a model with the corresponding annotations. A blank model does not provide them.

### NLTK

```python
import nltk
from markovify.nlp import NLTKNLP

nltk.download("punkt_tab")
nltk.download("stopwords")
nlp = NLTKNLP(language="spanish")
print(nlp.tokenize("Las proteínas interactúan."))
print(nlp.sentences("Primera oración. Segunda oración."))
print(nlp.remove_stopwords("Las proteínas interactúan."))
print(nlp.stem("proteínas interactuando"))
```

Resources are not downloaded automatically. Required resources can vary with NLTK version. tokenize forwards keyword arguments to word_tokenize; stem uses SnowballStemmer. stopwords returns the language-specific set.

For natural sentence generation, preserve punctuation, stopwords and inflection unless the experiment explicitly requires a different representation. Lemmatization and stemming are more suitable for lexical analysis.

## End-to-end workflow

```python
from markovify.nlp import Corpus, SpacyNLP, clean_text
from markovify import load_model

raw = Corpus.load("./library", formats=["pdf", "docx", "txt", "epub"],
                  errors="skip")
raw.save("raw.jsonl")
nlp = SpacyNLP(language="es")
processed = (raw.map(clean_text, urls=True, emails=True)
                .split(nlp.sentences)
                .filter_length(min_words=5, max_words=100)
                .deduplicate(case_sensitive=False))
if not len(processed):
    raise ValueError("No usable sentences extracted")

processed.save("processed.txt")
model = processed.to_model(state_size=3, random_state=42).compile()
model.save("model.h5", compression="gzip",
           metadata={"language": "es", "n_sentences": len(processed),
                     "source": "./library", "seed": 42})
restored = load_model("model.h5")
for sentence in restored.make_sentences(10, include=["proteína"],
                                        tries=100, max_attempts=200):
    print(sentence)
```

The workflow uses whitespace tokenization by default to preserve attached punctuation. A custom tokenizer can be supplied to to_model; the default word_join still joins tokens with spaces.

## Constrained text generation

```python
model.make_sentence(start_with="La ciencia", tries=100)
model.make_sentence(end_with="evidencia experimental.")
model.make_sentence(include=["proteína", "ligando"], match="all")
model.make_sentence(include=["proteína", "receptor"], match="any")
model.make_sentence(include_phrase="dinámica molecular")
model.make_sentence(exclude=["anticuerpo"])
model.make_sentence_around("dinámica molecular", tries=1000)
model.make_sentences(n=20, unique=True, max_attempts=500,
                     min_words=5, max_words=30, tries=100)
```

Constraints can be combined. Prefixes longer than state_size are supported by start_with if every forced transition has positive support. Other constraints use bounded rejection sampling. Matching is exact and case-sensitive at token level, including punctuation. make_sentence_around does not implement a reverse chain. Batch generation may return fewer than n results.

The original make_sentence_with_start(beginning, strict=True, **kwargs) remains available for compatibility, with its original state-size restriction. Prefer start_with for validated prefixes.

Generation controls include tries (default 10), min_words, max_words, max_steps, test_output, max_overlap_ratio (default 0.7), and max_overlap_total (default 15). The default chain generation limit is 10000 tokens; a direct Chain walk exceeding it raises GenerationLimitError.

## Model persistence

```python
from markovify import save_model, load_model

model.save("model.json", metadata={"dataset": "corpus-v1"})
model.save("model.json.gz")
model.save("model.h5")
model.save("model.hdf5", compression="gzip")

restored = load_model("model.json.gz")
print(restored.metadata)
save_model(restored, "copy.json")
```

Chain.load, Text.load and NewlineText.load also support class-checked loading. File format is selected by extension. HDF5 stores the versioned JSON envelope as a byte dataset, optionally compressed with gzip.

The envelope includes the model class, schema version, package version, UTC save time, user metadata, model data, RNG state and SHA-256 checksum. Writes use a temporary file and atomic replacement. The checksum detects corruption, not malicious tampering. Metadata must be JSON-serializable; supply it explicitly when saving a new model.

### Reproducible continuation

```python
from markovify import Chain, load_model

chain = Chain([["A", "B"], ["A", "C"]], state_size=1, random_state=42)
chain.walk()
chain.save("checkpoint.json")
expected = [chain.walk() for _ in range(20)]
restored = load_model("checkpoint.json")
assert expected == [restored.walk() for _ in range(20)]

# Start a different random stream:
restored = load_model("checkpoint.json", random_state=123)
```

The lower-level to_json/from_json methods remain available, as do Text.to_dict/from_dict. They do not preserve the complete file envelope or RNG state. Historical JSON can be loaded, but missing metadata, checksums and RNG state cannot be reconstructed. Historical text JSON does not distinguish Text from NewlineText; use NewlineText.load when the original class is known. Older releases may not read the new serialization format.

## General Markov chains

```python
from markovify import Chain

chain = Chain([["closed", "open", "closed"], ["closed", "open"]],
              state_size=1, random_state=42, max_steps=1000)
print(chain.walk())
print(list(chain.gen()))
print(chain.model)
```

The chain represents discrete sequences with beginning and ending markers. It is not a continuous-time model or a validated molecular Markov state model. Supported serialized tokens include strings, integers, finite floats, booleans, None and tuples of supported tokens.

## Combining and compiling models

```python
import markovify

a = markovify.NewlineText("A protein binds.\nA protein folds.", state_size=2)
b = markovify.NewlineText("A protein moves.\nA protein unfolds.", state_size=2)
combined = markovify.combine([a, b], weights=[1.5, 1], random_state=42)
compiled = combined.compile()
```

Models must have matching types and state sizes and be uncompiled when combined. Weights must be finite, nonnegative and not all zero. They scale transition counts, not normalized mixture probabilities. compile(inplace=True) modifies the existing model; otherwise compile returns a compiled model.

## Tutorials

See [notebooks/](notebooks/) for seven sequential tutorials covering extraction, NLP, EDA, training, generation, persistence and discrete chains. Install with `python -m pip install -e ".[all]"` and launch `jupyter lab notebooks`.

## Customization and development

Text supports retain_original=False to avoid retaining the corpus for overlap checks, well_formed=False to disable the default input rejection filter, and reject_reg for a custom rejection expression. Subclasses can override sentence_split, sentence_join, word_split, word_join, test_sentence_input and test_sentence_output.

```text
markovify/
  chain.py          Discrete sequence model
  text.py           Text and NewlineText
  constraints.py    Lexical constraints
  validation.py     Parameter validation and RNG
  serialization.py  Typed chain serialization
  io.py             File persistence
  utils.py          Model combination
  nlp/
    cleaning.py     Text cleaning
    backends.py     spaCy and NLTK wrappers
    corpus.py       Corpus operations
    documents.py    Document readers and discovery
test/               Regression tests
pyproject.toml      Packaging and optional dependencies
```

Run tests and build distributions from the repository root:

```bash
python -m pip install -e ".[all]"
python -m pytest
python -m build
```

Optional backend tests may skip when dependencies are absent. The CI workflow and current results are linked at the top of this README; this documentation does not assert that every optional backend or Python version has passed.

## Current limitations

- PDF extraction requires a text layer; OCR, complex layout reconstruction, header/footer removal and bibliography detection are not implemented.
- DOCX paragraphs and tables are extracted separately, without preserving their relative order. Legacy DOC, PPTX and XLSX are not supported.
- Corpus processing is in memory. Source provenance is not automatically retained after transformations or in saved corpus files.
- NLP models and resources must be installed separately; dependency versions are not locked.
- Lexical constraints other than prefixes use rejection sampling; rare constraints may be inefficient.
- HDF5 does not expose native transition matrices or partial model loading.
- Automatic persistence supports Chain, Text and NewlineText, not arbitrary custom subclasses.
- Generated text can reproduce training material and is not guaranteed to be factual or grammatically correct.

## License and attribution

MIT License. See [LICENSE](LICENSE). Original Markovify by Jeremy Singer-Vine, initially developed at BuzzFeed. This repository contains subsequent extensions. The historical project examples and contributor acknowledgments below are retained from the original README; external links may no longer be active.

- [@aogier](https://github.com/aogier)
- [@terisikk](https://github.com/terisikk)

Initially developed at [BuzzFeed](https://www.buzzfeed.com).
