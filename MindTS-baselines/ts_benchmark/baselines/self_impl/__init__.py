from importlib import import_module

__all__ = [
    "AnomalyTransformer",
    "GDN",
    "InterFusion",
    "MTADGAT",
    "OmniAnomaly",
    "TranAD",
    "USAD",
]


_MODEL_IMPORTS = {
    "AnomalyTransformer": (
        "ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer",
        "AnomalyTransformer",
    ),
    "GDN": ("ts_benchmark.baselines.self_impl.GDN.GDN", "GDN"),
    "InterFusion": ("ts_benchmark.baselines.self_impl.InterFusion.InterFusion", "InterFusion"),
    "MTADGAT": ("ts_benchmark.baselines.self_impl.MTADGAT.MTADGAT", "MTADGAT"),
    "OmniAnomaly": ("ts_benchmark.baselines.self_impl.OmniAnomaly.OmniAnomaly", "OmniAnomaly"),
    "TranAD": ("ts_benchmark.baselines.self_impl.TranAD.TranAD", "TranAD"),
    "USAD": ("ts_benchmark.baselines.self_impl.USAD.USAD", "USAD"),
}


def __getattr__(name):
    """Import optional baselines only when their model is requested."""
    try:
        module_name, attribute_name = _MODEL_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
