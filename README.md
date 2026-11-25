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

## Markovify In The Wild

- BuzzFeed's [Tom Friedman Sentence Generator](http://www.buzzfeed.com/jsvine/the-tom-friedman-sentence-generator) / [@mot_namdeirf](https://twitter.com/mot_namdeirf).
- [/u/user_simulator](https://www.reddit.com/user/user_simulator), a Reddit bot that generates comments based on a user's comment history. [[code](https://github.com/trambelus/UserSim)]
- [SubredditSimulator](https://www.reddit.com/r/SubredditSimulator), which [uses `markovify`](https://www.reddit.com/r/SubredditSimMeta/comments/3d910r/i_was_inspired_by_this_place_and_made_a_twitter/ct3vjp0) to generate random Reddit submissions and comments based on a subreddit's previous activity. [[code](https://github.com/Deimos/SubredditSimulator)]
- [college crapplication](http://college-crapplication.appspot.com/), a web-app that generates college application essays. [[code](https://github.com/mattr555/college-crapplication)]
- [@MarkovPicard](https://twitter.com/MarkovPicard), a Twitter bot based on *Star Trek: The Next Generation* transcripts. [[code](https://github.com/rdsheppard95/MarkovPicard)]
- [sekrits.herokuapp.com](https://sekrits.herokuapp.com/), a `markovify`-powered quiz that challenges you to tell the difference between "two file titles relating to matters of [Australian] national security" — one real and one fake. [[code](https://sekrits.herokuapp.com/)]
- [Hacker News Simulator](http://news.ycombniator.com/), which does what it says on the tin. [[code](https://github.com/orf/hnewssimulator)]
- [Stak Attak](http://www.stakattak.me/), a "poetic stackoverflow answer generator." [[code](https://github.com/theannielin/hackharvard)]
- [MashBOT](https://twitter.com/mashomatic), a `markovify`-powered Twitter bot attached to a printer. Presented by [Helen J Burgess at Babel Toronto 2015](http://electric.press/mash/). [[code](https://github.com/hyperrhiz/mashbot)]
- [The Mansfield Reporter](http://maxlupo.com/mansfield-reporter/), "a simple device which can generate new text from some of history's greatest authors [...] running on a tiny Raspberry Pi, displaying through a tft screen from Adafruit." 
- [twitter markov](https://github.com/fitnr/twitter_markov), a tool to "create markov chain ("_ebooks") accounts on Twitter."
- [@Bern_Trump_Bot](https://twitter.com/bern_trump_bot), "Bernie Sanders and Donald Trump driven by Markov Chains." [[code](https://github.com/MichaelMartinez/Bern_Trump_Bot)]
- [@RealTrumpTalk](https://twitter.com/RealTrumpTalk), "A bot that uses the things that @realDonaldTrump tweets to create it's own tweets." [[code](https://github.com/CastleCorp/TrumpTalk)]
- [Taylor Swift Song Generator](http://taytay.mlavin.org/), which does what it says. [[code](https://github.com/caktus/taytay)]
- [@BOTtalks](https://twitter.com/bottalks) / [ideasworthautomating.com](http://ideasworthautomating.com/). "TIM generates talks on a broad spectrum of topics, based on the texts of slightly more coherent talks given under the auspices of his more famous big brother, who shall not be named here." [[code](https://github.com/alexislloyd/tedbot)]
- [Internal Security Zones](http://rebecca-ricks.com/2016/05/06/internal-security-zones/), "Generative instructions for prison design & maintenance." [[code](https://github.com/baricks/internal-security-zones)]
- [Miraculous Ladybot](http://miraculousladybot.tumblr.com/). Generates [Miraculous Ladybug](https://en.wikipedia.org/wiki/Miraculous:_Tales_of_Ladybug_%26_Cat_Noir) fanfictions and posts them on Tumblr. [[code](https://github.com/veggiedefender/miraculousladybot)]
- [@HaikuBotto](https://twitter.com/HaikuBotto), "I'm a bot that writes haiku from literature. beep boop" [[code](https://github.com/balysv/HaikuBotto)]
- [Chat Simulator Bot](http://www.telegram.me/ChatSimulatorBot), a bot for Telegram. [[code](https://github.com/GuyAglionby/chatsimulatorbot)]
- [emojipasta.club](http://emojipasta.club), "a web service that exposes RESTful endpoints for generating emojipastas, as well as a simple frontend for generating and tweeting emojipasta sentences." [[code](https://github.com/ntratcliff/emojipasta.club)]
- [Towel Generator](http://towel.labs.wasv.me/), "A system for generating sentences similar to those from the hitchhikers series of books." [[code](https://github.com/wastevensv/towelday)]
- [@mercurialbot](https://twitter.com/mercurialbot), "A twitter bot that generates tweets based on its mood." [[code](https://github.com/brahmcapoor/Mercury)]
- [becomeacurator.com](http://becomeacurator.com/), which "generates curatorial statements for contemporary art expositions, using Markov chains and texts from galleries around the world." [[code](https://github.com/jjcastro/markov-curatorial-generator)]
- [mannynotfound/interview-bot](https://github.com/mannynotfound/interview-bot), "A python based terminal prompt app to automate the interview process."
- [Steam Game Generator](http://applepinegames.com/tech/steam-game-generator), which "uses data from real Steam games, randomized using Markov chains." [[code](https://github.com/applepinegames/steam_game_generator)]
- [@DicedOnionBot](https://twitter.com/DicedOnionBot), which "generates new headlines by The Onion by regurgitating and combining old headlines." [[code](https://github.com/mobeets/fake-onion)]
- [@thought__leader](https://twitter.com/thought__leader), "Thinking thoughts so you don't have to!" [[blog post](http://jordan-wright.com/blog/post/2016-04-08-i-automated-infosec-thought-leadership/)]
- [@_murakamibot](https://twitter.com/_murakamibot) and [@jamesjoycebot](https://twitter.com/jamesjoycebot), bots that tweet Haruki Murakami and James Joyce-like sentences. [[code](https://github.com/tmkuba/markovBot)]
- [shartificialintelligence.com](http://www.shartificialintelligence.com/), "the world's first creative ad agency staffed entirely with copywriter robots." [[code](https://github.com/LesGuessing/shartificial-intelligence)]
- [@NightValeFeed](https://twitter.com/NightValeFeed), which "generates tweets by combining [@NightValeRadio](https://twitter.com/NightValeRadio) tweets with [@BuzzFeed](https://twitter.com/BuzzFeed) headlines." [[code](https://github.com/stepjue/night-vale-buzzfeed)]
- [Wynbot9000](https://github.com/ammgws/wynbot), which "mimics your friends on Google Hangouts." [[code](https://github.com/ammgws/wynbot)]
- [@sealDonaldTrump](https://twitter.com/sealdonaldtrump), "a twitter bot that sounds like @realDonaldTrump, with an aquatic twist." [[code](https://github.com/lukewrites/sealdonaldtrump)]
- [@veeceebot](https://twitter.com/veeceebot), which is "like VCs but better!" [[code](https://github.com/yasyf/vcbot)]
- [@mar_phil_bot](https://twitter.com/mar_phil_bot), a Twitter bot [trained](http://gfleetwood.github.io/philosophy-bot/) on Nietzsche, Russell, Kant, Machiavelli, and Plato. [[code](https://gist.github.com/gfleetwood/569804c4f2ab372746661996542a8065)]
- [funzo-facts](https://github.com/smalawi/funzo-facts), a program that generates never-before-seen trivia based on Jeopardy! questions. [[code](https://github.com/smalawi/funzo-facts/blob/master/funzo_fact_gen.py)]
- [Chains Invent Insanity](http://chainsinventinsanity.com), a [Cards Against Humanity](https://cardsagainsthumanity.com) answer card generator. [[code](https://github.com/TuxOtaku/chains-invent-insanity)]
- [@CanDennisDream](https://twitter.com/CanDennisDream), a twitter bot that contemplates life by training on existential literature discussions. [[code](https://github.com/GiantsLoveDeathMetal/dennis_bot)]
- [B-9 Indifference](https://github.com/eoinnoble/b9-indifference), a program that generates a _Star Trek: The Next Generation_ script of arbitrary length using Markov chains trained on the show’s episode and movie scripts. [[code](https://github.com/eoinnoble/b9-indifference)]
- [adam](http://bziarkowski.pl/adam), polish poetry generator. [[code](https://github.com/bziarkowski/adam)]
- [Stackexchange Simulator](https://se-simulator.lw1.at/), which uses StackExchange's bulk data to generate random questions and answers. [[code](https://github.com/Findus23/se-simulator)]
- [@BloggingBot](https://twitter.com/BloggingBot), tweets sentences based on a corpus of 17 years of [blogging](http://artlung.com/blog/2018/02/23/markov-chains-are-hilarious/).
- [Commencement Speech Generator](https://github.com/whatrocks/markov-commencement-speech), generates "graduation speech"-style quotes from a dataset of the "greatest of all time" commencement speeches)
- [@alg_testament](https://twitter.com/alg_testament), tweets sentences based on The Old Testament and two coding textbooks in Russian. [[code](https://github.com/maryszmary/Algorithm-Testament)]  
- [@IRAMockBot](https://twitter.com/IRAMockBot), uses Twitter's data on tweets from Russian IRA-associated accounts to produce fake IRA tweets, for educational and study purposes.[[code](https://github.com/nwithan8/IRAMockBot)]
- [Personal Whatsapp Chat Analyzer](https://github.com/Giuzzilla/Personal-Whatsapp-Chat-Analyzer), some basic analytics for WhatsApp chat exports (private & groups), word counting & markov chain phrase generator 
- [DeepfakeBot](https://deepfake-bot.readthedocs.io/), a system for converting your friends into Discord bots. [[code](https://github.com/rustygentile/deepfake-bot)]
- [python-markov-novel](https://github.com/accraze/python-markov-novel), writes a random novel using markov chains, broken down into chapters
- [python-ia-markov](https://github.com/accraze/python-ia-markov), trains Markov models on Internet Archive text files
- [@bot_homer](https://twitter.com/bot_homer), a Twitter bot trained using Homer Simpson's dialogues of 600 chapters. [[code](https://github.com/ivanlen/simpsons_bot)].
- [git-commit-gen](https://github.com/solean/git-commit-gen), generates git commit messages by using markovify to build a model of a repo's git log
- [fakesocial](https://fakesocial.net), a fake social network using generated content. [[code](https://github.com/berfr/fakesocial)]
- [Slovodel Bot](https://github.com/weiss-d/slovodel-bot), a Telegram bot that generates non-existent Russian words using corpus made by algorithmically dividing existent words into syllables.
- [Deuterium](https://github.com/portasynthinca3/deuterium), a Discord bot that generates messages on its own, after analyzing yours, and learning constantly. There's also a global model shared with all other servers.
- [Markovify Piano](https://github.com/asigalov61/Markovify-Piano), generates coherent and plausible music generation.
- [TweetyPy](https://twitter.com/_TweetyPy_), a Twitter bot that takes data from the top US trends or user tweets, learn and create own tweets and word clouds. [[code](https://github.com/SerhiiStets/TweetyPy)]
- [cappuccino/ai.py](https://github.com/TheReverend403/cappuccino), an IRC bot with a plugin that generates sentences based on the PostgreSQL-stored logs of the IRC channels it's in. [[code](https://github.com/TheReverend403/cappuccino/blob/main/cappuccino/ai.py)]
- [django-markov](https://github.com/andrlik/django-markov), a reusable Django app that provides a generic backend to generate and store Markov chains for later retrieval and generation of sentences.

Have other examples? Pull requests welcome.

## Thanks

Many thanks to the following GitHub users for contributing code and/or ideas:

- [@orf](https://github.com/orf)
- [@deimos](https://github.com/deimos)
- [@cjmochrie](https://github.com/cjmochrie)
- [@Jaza](https://github.com/Jaza)
- [@fitnr](https://github.com/fitnr)
- [@andela-mfalade](https://github.com/andela-mfalade)
- [@ntratcliff](https://github.com/ntratcliff)
- [@schollz](https://github.com/schollz)
- [@aalireza](https://github.com/aalireza)
- [@bfontaine](https://github.com/bfontaine)
- [@tmsherman](https://github.com/tmsherman)
- [@wodim](https://github.com/wodim)
- [@eh11fx](https://github.com/eh11fx)
- [@ammgws](https://github.com/ammgws)
- [@OtakuMegane](https://github.com/OtakuMegane)
- [@tsunaminoai](https://github.com/tsunaminoai)
- [@MatthewScholefield](https://github.com/MatthewScholefield)
- [@danmayer](https://github.com/danmayer)
- [@kade-robertson](https://github.com/kade-robertson)
- [@erikerlandson](https://github.com/erikerlandson)
- [@briennakh](https://github.com/briennakh)
- [@berfr](https://github.com/berfr)
- [@Freestackmejai](https://github.com/Freestackmejai)
- [@rokala](https://github.com/rokala)
- [@eumiro](https://github.com/eumiro)
- [@monosans](https://github.com/monosans)
- [@aogier](https://github.com/aogier)
- [@terisikk](https://github.com/terisikk)

Initially developed at [BuzzFeed](https://www.buzzfeed.com).
