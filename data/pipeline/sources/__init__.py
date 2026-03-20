from __future__ import annotations
from .base import BaseSource

SOURCES: dict[str, BaseSource] = {}


def register_sources(ontology) -> dict[str, BaseSource]:
    from .fsd50k import FSD50KSource
    from .esc50 import ESC50Source
    from .disco import DISCOSource
    from .cipic import CIPICSource
    from .musdb18 import MUSDB18Source
    from .tau import TAUSource

    sources = {
        "fsd50k": FSD50KSource(ontology),
        "esc50": ESC50Source(ontology),
        "disco": DISCOSource(ontology),
        "cipic": CIPICSource(),
        "musdb18": MUSDB18Source(ontology),
        "tau": TAUSource(),
    }
    SOURCES.update(sources)
    return sources
