__all__ = [
    "PairCopyConfig",
    "PairCopyDataset",
    "BoundedImageOrganism",
    "OrganismConfig",
    "make_variant",
]


def __getattr__(name):
    """Keep scalar-repair utilities importable without the optional Torch lab."""

    if name in {"PairCopyConfig", "PairCopyDataset"}:
        from .data import PairCopyConfig, PairCopyDataset

        return {"PairCopyConfig": PairCopyConfig, "PairCopyDataset": PairCopyDataset}[name]
    if name in {"BoundedImageOrganism", "OrganismConfig", "make_variant"}:
        from .model import BoundedImageOrganism, OrganismConfig, make_variant

        return {
            "BoundedImageOrganism": BoundedImageOrganism,
            "OrganismConfig": OrganismConfig,
            "make_variant": make_variant,
        }[name]
    raise AttributeError(name)
