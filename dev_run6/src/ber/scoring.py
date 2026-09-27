"""Exact re-implementation of the challenge metric: per-S1 F0.5, macro-averaged, singletons included."""


def f05_entity(pred, true):
    """F0.5 for one Source-1 entity. Empty truth: 1.0 only for an empty prediction."""
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    fp, fn = len(pred) - tp, len(true) - tp
    return 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)


def macro_f05(pred_map, true_map):
    """Macro F0.5 over every key of true_map (missing predictions count as empty)."""
    if not true_map:
        return float("nan")
    return sum(f05_entity(set(pred_map.get(k, ())), set(v)) for k, v in true_map.items()) / len(true_map)
