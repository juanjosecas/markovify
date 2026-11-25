# Notebooks

From the repository root:

```bash
python -m pip install -e ".[all]"
jupyter lab notebooks
```

Run notebooks 01–06 in order; 07 is independent. Run from the repository root or notebooks directory. The default synthetic corpus is illustrative, not scientific evidence. Set SOURCE and FORMATS in notebook 01 for your own documents. Outputs are written to notebooks/outputs and overwritten on reruns. Optional lemmatization requires es_core_news_sm. For real data, consider document-level splits to avoid leakage; the demonstration uses a sentence-level holdout.

Notebook 04 reports transition coverage and conditional NLL separately; unseen transitions imply infinite unsmoothed perplexity. Order 2 is fixed for illustration. Notebook 05 disables novelty filtering for the small demo. Notebook 06 benchmarks persistence formats, including HDF5. All notebooks are committed without execution outputs.
