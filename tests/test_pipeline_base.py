import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def test_ontology_loads():
    from data.pipeline.ontology import Ontology
    ont = Ontology(str(DATA_DIR / "ontology.json"))
    assert ont.get_label("/m/09x0r") == "Speech"


def test_ontology_subtree():
    from data.pipeline.ontology import Ontology
    ont = Ontology(str(DATA_DIR / "ontology.json"))
    subtree = ont.get_subtree("/m/0dgw9r")  # Human sounds
    assert len(subtree) > 1


def test_base_source_abc():
    from data.pipeline.sources.base import BaseSource
    with pytest.raises(TypeError):
        BaseSource()  # Cannot instantiate ABC
