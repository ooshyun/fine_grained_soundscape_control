from __future__ import annotations

import json


class Ontology:
    """AudioSet ontology wrapper for navigating the class hierarchy.

    Parameters
    ----------
    json_path : str
        Path to the AudioSet ``ontology.json`` file.
    """

    # Top-level category IDs
    ANIMAL = "/m/0jbk"
    SOUNDS_OF_THINGS = "/t/dd00041"
    HUMAN_SOUNDS = "/m/0dgw9r"
    NATURAL_SOUNDS = "/m/059j3w"
    SOURCE_AMBIGUOUS_SOUNDS = "/t/dd00098"
    CHANNEL_ENVIRONMENT_BACKGROUND = "/t/dd00123"
    MUSIC = "/m/04rlf"
    ROOT = "__root__"

    def __init__(self, json_path: str) -> None:
        with open(json_path, "rb") as f:
            ontology_list = json.load(f)

        root_node = {
            "child_ids": [
                self.SOURCE_AMBIGUOUS_SOUNDS,
                self.ANIMAL,
                self.SOUNDS_OF_THINGS,
                self.HUMAN_SOUNDS,
                self.NATURAL_SOUNDS,
                self.CHANNEL_ENVIRONMENT_BACKGROUND,
                self.MUSIC,
            ],
            "id": self.ROOT,
            "name": self.ROOT,
        }
        ontology_list.append(root_node)

        self.ontology: dict[str, dict] = {
            item["id"]: item for item in ontology_list
        }

        self._dfs()
        self._mark_source_ambiguous()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _dfs(self, node_id: str | None = None) -> None:
        if node_id is None:
            node_id = self.ROOT
            self.ontology[node_id]["depth"] = 0
            self.ontology[node_id]["parent_id"] = None
        else:
            parent_node = self.ontology[node_id]["parent_id"]
            self.ontology[node_id]["depth"] = (
                self.ontology[parent_node]["depth"] + 1
            )

        self.ontology[node_id]["source_ambiguous"] = 0

        for child_id in self.ontology[node_id]["child_ids"]:
            self.ontology[child_id]["parent_id"] = node_id
            self._dfs(node_id=child_id)

    def _mark_source_ambiguous(self, node_id: str | None = None) -> None:
        if node_id is None:
            node_id = self.SOURCE_AMBIGUOUS_SOUNDS

        self.ontology[node_id]["source_ambiguous"] = 1

        for child_id in self.ontology[node_id]["child_ids"]:
            self._mark_source_ambiguous(child_id)

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_label(self, node_id: str) -> str:
        """Return the human-readable name for *node_id*."""
        return self.ontology[node_id]["name"]

    def get_id_from_name(self, name: str) -> str | None:
        """Return the AudioSet ID for a class *name*, or None if not found."""
        for _id, node in self.ontology.items():
            if node["name"] == name:
                return _id
        return None

    def is_source_ambiguous(self, node_id: str) -> bool:
        return bool(self.ontology[node_id]["source_ambiguous"])

    def is_reachable(self, parent: str, child: str) -> bool:
        """Return ``True`` if *child* is reachable from *parent* in the tree."""
        assert parent in self.ontology, "parent not in ontology"
        assert child in self.ontology, "child not in ontology"

        if parent == child:
            return True
        for child_id in self.ontology[parent]["child_ids"]:
            if self.is_reachable(child_id, child):
                return True
        return False

    def get_subtree(self, node_id: str) -> list[str]:
        """Return all IDs reachable from *node_id* (inclusive)."""
        subtree = [node_id]
        for child_id in self.ontology[node_id]["child_ids"]:
            subtree.extend(self.get_subtree(child_id))
        return subtree

    def get_ancestor_ids(self, node_id: str) -> list[str]:
        """Return ancestor chain from root down to *node_id*."""
        assert node_id in self.ontology, "id not in ontology"

        ancestor_ids = [node_id]
        parent_id = self.ontology[node_id]["parent_id"]
        while parent_id is not None:
            ancestor_ids.append(parent_id)
            parent_id = self.ontology[parent_id]["parent_id"]
        return list(reversed(ancestor_ids))

    def is_leaf_node(self, node_id: str) -> bool:
        assert node_id in self.ontology, "id not in ontology"
        return len(self.ontology[node_id]["child_ids"]) == 0

    def unsmear(self, ids: list[str]) -> list[str]:
        """Remove ancestor duplicates -- keep only the deepest nodes."""
        ids_sorted = sorted(ids, key=lambda x: -self.ontology[x]["depth"])

        unsmeared: list[str] = []
        removed: list[int] = []
        for i in range(len(ids)):
            if i in removed:
                continue
            node_id = ids[i]
            unsmeared.append(node_id)
            while self.ontology[node_id]["parent_id"] is not None:
                node_id = self.ontology[node_id]["parent_id"]
                try:
                    idx = ids.index(node_id)
                    removed.append(idx)
                except ValueError:
                    pass

        return unsmeared
