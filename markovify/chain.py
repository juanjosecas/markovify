import operator
import bisect
import json
import copy
from .io import PersistenceMixin
from .serialization import encode_model, decode_model
from .validation import positive_int, make_rng


class GenerationLimitError(RuntimeError):
    pass

BEGIN = "___BEGIN__"
END = "___END__"


def accumulate(iterable, func=operator.add):
    """
    Cumulative calculations. (Summation, by default.)
    Via: https://docs.python.org/3/library/itertools.html#itertools.accumulate
    """
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def compile_next(next_dict):
    words = list(next_dict.keys())
    cff = list(accumulate(next_dict.values()))
    return [words, cff]


class Chain(PersistenceMixin):
    """
    A Markov chain representing processes that have both beginnings and ends.
    For example: Sentences.
    """

    def __init__(self, corpus, state_size, model=None, *, random_state=None, max_steps=10000):
        """
        `corpus`: A list of lists, where each outer list is a "run"
        of the process (e.g., a single sentence), and each inner list
        contains the steps (e.g., words) in the run. If you want to simulate
        an infinite process, you can come very close by passing just one, very
        long run.

        `state_size`: An integer indicating the number of items the model
        uses to represent its state. For text generation, 2 or 3 are typical.
        """
        self.state_size = positive_int(state_size, 'state_size')
        self.max_steps = positive_int(max_steps, 'max_steps')
        self.rng = make_rng(random_state)
        self.model = self.build(corpus, state_size) if model is None else model
        if not self.model or (BEGIN,) * state_size not in self.model:
            raise ValueError('Model must contain a beginning state')
        self.compiled = (len(self.model) > 0) and (
            type(self.model[tuple([BEGIN] * state_size)]) == list
        )
        if not self.compiled:
            self.precompute_begin_state()

    def compile(self, inplace=False):
        if self.compiled:
            if inplace:
                return self
            return Chain(None, self.state_size, model=copy.deepcopy(self.model), random_state=copy.deepcopy(self.rng), max_steps=self.max_steps)
        mdict = {
            state: compile_next(next_dict) for (state, next_dict) in self.model.items()
        }
        if not inplace:
            return Chain(None, self.state_size, model=mdict, random_state=copy.deepcopy(self.rng), max_steps=self.max_steps)
        self.model = mdict
        self.compiled = True
        return self

    def build(self, corpus, state_size):
        """
        Build a Python representation of the Markov model. Returns a dict
        of dicts where the keys of the outer dict represent all possible states,
        and point to the inner dicts. The inner dicts represent all possibilities
        for the "next" item in the chain, along with the count of times it
        appears.
        """

        # Using a DefaultDict here would be a lot more convenient, however the memory
        # usage is far higher.
        model = {}

        if corpus is None or isinstance(corpus, (str, bytes)):
            raise ValueError('Expected a corpus of token sequences')
        for run in corpus:
            if isinstance(run, (str, bytes)):
                raise ValueError('Each run must be a token sequence')
            run = list(run)
            if any(token in (BEGIN, END) for token in run):
                raise ValueError('Reserved token in corpus')
            items = ([BEGIN] * state_size) + run + [END]
            for i in range(len(run) + 1):
                state = tuple(items[i : i + state_size])
                follow = items[i + state_size]
                if state not in model:
                    model[state] = {}

                if follow not in model[state]:
                    model[state][follow] = 0

                model[state][follow] += 1
        return model

    def precompute_begin_state(self):
        """
        Caches the summation calculation and available choices for BEGIN * state_size.
        Significantly speeds up chain generation on large corpora. Thanks, @schollz!
        """
        begin_state = tuple([BEGIN] * self.state_size)
        choices, cumdist = compile_next(self.model[begin_state])
        self.begin_cumdist = cumdist
        self.begin_choices = choices

    def move(self, state):
        """
        Given a state, choose the next item at random.
        """
        if self.compiled:
            choices, cumdist = self.model[state]
        elif state == tuple([BEGIN] * self.state_size):
            choices = self.begin_choices
            cumdist = self.begin_cumdist
        else:
            choices, weights = zip(*self.model[state].items())
            cumdist = list(accumulate(weights))
        r = self.rng.random() * cumdist[-1]
        selection = choices[bisect.bisect(cumdist, r)]
        return selection

    def gen(self, init_state=None, *, max_steps=None):
        """
        Starting either with a naive BEGIN state, or the provided `init_state`
        (as a tuple), return a generator that will yield successive items
        until the chain reaches the END state.
        """
        state = init_state or (BEGIN,) * self.state_size
        limit = self.max_steps if max_steps is None else positive_int(max_steps, 'max_steps', allow_zero=True)
        for step in range(limit + 1):
            next_word = self.move(state)
            if next_word == END:
                break
            if step == limit:
                raise GenerationLimitError('Generation exceeded max_steps')
            yield next_word
            state = tuple(state[1:]) + (next_word,)

    def walk(self, init_state=None, *, max_steps=None):
        """
        Return a list representing a single run of the Markov model, either
        starting with a naive BEGIN state, or the provided `init_state`
        (as a tuple).
        """
        return list(self.gen(init_state, max_steps=max_steps))

    def to_json(self):
        """
        Dump the model as a JSON object, for loading later.
        """
        return json.dumps(encode_model(self), allow_nan=False)

    @classmethod
    def from_json(cls, json_thing, **kwargs):
        """
        Given a JSON object or JSON string that was created by `self.to_json`,
        return the corresponding markovify.Chain.
        """

        obj = json.loads(json_thing) if isinstance(json_thing, str) else json_thing

        if isinstance(obj, dict) and obj.get('format') == 'markovify.chain':
            model, state_size, max_steps = decode_model(obj)
            kwargs.setdefault('max_steps', max_steps)
            return cls(None, state_size, model, **kwargs)
        if isinstance(obj, list):
            rehydrated = {tuple(item[0]): item[1] for item in obj}
        elif isinstance(obj, dict):
            rehydrated = obj
        else:
            raise ValueError("Object should be dict or list")

        if not rehydrated:
            raise ValueError('Empty model')
        state_size = len(next(iter(rehydrated)))

        inst = cls(None, state_size, rehydrated, **kwargs)
        return inst
