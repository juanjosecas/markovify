from math import isfinite
from numbers import Real
from .chain import Chain
from .text import Text


def get_model_dict(thing):
    if isinstance(thing, Chain):
        if thing.compiled:
            raise ValueError("Not implemented for compiled markovify.Chain")
        return thing.model
    if isinstance(thing, Text):
        if thing.chain.compiled:
            raise ValueError("Not implemented for compiled markovify.Chain")
        return thing.chain.model
    if isinstance(thing, list):
        return dict(thing)
    if isinstance(thing, dict):
        return thing

    raise ValueError(
        "`models` should be instances of list, dict, markovify.Chain, or markovify.Text"
    )


def combine(models, weights=None, *, random_state=None):
    models = list(models)
    if not models:
        raise ValueError('At least one model is required')
    if weights is None:
        weights = [1 for _ in range(len(models))]

    if len(models) != len(weights):
        raise ValueError("`models` and `weights` lengths must be equal.")

    if any(isinstance(w, bool) or not isinstance(w, Real) or not isfinite(w) or w < 0 for w in weights):
        raise ValueError('Weights must be finite nonnegative numbers')
    if not any(w > 0 for w in weights):
        raise ValueError('At least one weight must be positive')
    model_dicts = list(map(get_model_dict, models))
    if any(not m for m in model_dicts):
        raise ValueError('Empty model')
    state_sizes = [len(list(md.keys())[0]) for md in model_dicts]

    if len(set(state_sizes)) != 1:
        raise ValueError("All `models` must have the same state size.")

    if len(set(map(type, models))) != 1:
        raise ValueError("All `models` must be of the same type.")

    c = {}

    for m, w in zip(model_dicts, weights):
        if w == 0:
            continue
        for state, options in m.items():
            current = c.get(state, {})
            for subseq_k, subseq_v in options.items():
                subseq_prev = current.get(subseq_k, 0)
                current[subseq_k] = subseq_prev + (subseq_v * w)
            c[state] = current

    ret_inst = models[0]

    if isinstance(ret_inst, Chain):
        return Chain.from_json(c, random_state=random_state)
    if isinstance(ret_inst, Text):
        ret_inst.find_init_states_from_chain.cache_clear()
        if any(m.retain_original and w > 0 for m, w in zip(models, weights)):
            combined_sentences = []
            for m, w in zip(models, weights):
                if w > 0 and m.retain_original:
                    combined_sentences += m.parsed_sentences
            return ret_inst.from_chain(c, parsed_sentences=combined_sentences, random_state=random_state)
        else:
            return ret_inst.from_chain(c, random_state=random_state)
    if isinstance(ret_inst, list):
        return list(c.items())
    if isinstance(ret_inst, dict):
        return c
