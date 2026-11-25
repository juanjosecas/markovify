import math


def encode_token(token):
    if isinstance(token, tuple):
        return {"tuple": [encode_token(item) for item in token]}
    if token is None or type(token) in (str, int, bool):
        return token
    if type(token) is float and math.isfinite(token):
        return token
    raise TypeError("Unsupported JSON token type")


def decode_token(token):
    if isinstance(token, dict) and set(token) == {"tuple"}:
        return tuple(decode_token(item) for item in token["tuple"])
    if token is None or type(token) in (str, int, bool, float):
        return token
    raise ValueError("Invalid serialized token")


def encode_model(chain):
    rows = []
    for state, options in chain.model.items():
        if chain.compiled:
            choices, cumulative = options
            values = [[encode_token(t), w] for t, w in zip(choices, cumulative)]
        else:
            values = [[encode_token(t), w] for t, w in options.items()]
        rows.append([[encode_token(t) for t in state], values])
    return dict(format="markovify.chain", version=1, state_size=chain.state_size,
                max_steps=chain.max_steps, compiled=chain.compiled, rows=rows)


def decode_model(obj):
    if obj["version"] != 1:
        raise ValueError("Unsupported chain format version")
    model = {}
    for state, values in obj["rows"]:
        key = tuple(decode_token(t) for t in state)
        pairs = [(decode_token(t), w) for t, w in values]
        if obj["compiled"]:
            model[key] = [[t for t, w in pairs], [w for t, w in pairs]]
        else:
            model[key] = dict(pairs)
    return model, obj["state_size"], obj["max_steps"]
