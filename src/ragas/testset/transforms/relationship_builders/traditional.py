import typing as t
from collections import Counter
from dataclasses import dataclass

from ragas.metrics._string import DistanceMeasure
from ragas.testset.graph import KnowledgeGraph, Node, Relationship
from ragas.testset.transforms.base import RelationshipBuilder
from ragas.testset.transforms.relationship_builders.cosine import (
    _nodes_are_not_siblings,
)


@dataclass
class JaccardSimilarityBuilder(RelationshipBuilder):
    property_name: str = "entities"
    key_name: t.Optional[str] = None
    new_property_name: str = "jaccard_similarity"
    threshold: float = 0.5

    def _jaccard_similarity(self, set1: t.Set[str], set2: t.Set[str]) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        if self.property_name is None:
            self.property_name

        similar_pairs = []
        for i, node1 in enumerate(kg.nodes):
            for j, node2 in enumerate(kg.nodes):
                if i >= j:
                    continue
                items1 = node1.get_property(self.property_name)
                items2 = node2.get_property(self.property_name)
                if items1 is None or items2 is None:
                    raise ValueError(
                        f"Node {node1.id} or {node2.id} has no {self.property_name}"
                    )
                if self.key_name is not None:
                    items1 = items1.get(self.key_name, [])
                    items2 = items2.get(self.key_name, [])
                similarity = self._jaccard_similarity(set(items1), set(items2))
                if similarity >= self.threshold:
                    similar_pairs.append((i, j, similarity))

        return [
            Relationship(
                source=kg.nodes[i],
                target=kg.nodes[j],
                type="jaccard_similarity",
                properties={self.new_property_name: similarity_float},
                bidirectional=True,
            )
            for i, j, similarity_float in similar_pairs
        ]


@dataclass
class OverlapScoreBuilder(RelationshipBuilder):
    property_name: str = "entities"
    key_name: t.Optional[str] = None
    new_property_name: str = "overlap_score"
    distance_measure: DistanceMeasure = DistanceMeasure.JARO_WINKLER
    distance_threshold: float = 0.9
    noise_threshold: float = 0.05
    threshold: float = 0.01
    target_cross_source: t.Optional[t.Callable[[Node], bool]] = None

    def __post_init__(self):
        try:
            from rapidfuzz import distance
        except ImportError:
            raise ImportError(
                "rapidfuzz is required for string distance. Please install it using `pip install rapidfuzz`"
            )

        self.distance_measure_map = {
            DistanceMeasure.LEVENSHTEIN: distance.Levenshtein,
            DistanceMeasure.HAMMING: distance.Hamming,
            DistanceMeasure.JARO: distance.Jaro,
            DistanceMeasure.JARO_WINKLER: distance.JaroWinkler,
        }

    def _overlap_score(
        self, matched_x: int, matched_y: int, total_x: int, total_y: int
    ) -> float:
        """
        Calculate a modified F1 score to measure overlap between two sets.

        Args:
            matched_x: Set of indices from set X that have matches
            matched_y: Set of indices from set Y that have matches
            total_x: Total number of items in set X (excluding noisy items)
            total_y: Total number of items in set Y (excluding noisy items)

        Returns:
            float: Overlap score between 0.0 and 1.0
        """
        if total_x == 0 or total_y == 0 or matched_x == 0 and matched_y == 0:
            return 0.0

        match_ratio_x = matched_x / total_x
        match_ratio_y = matched_y / total_y

        # F1-style score: 2 * (match_ratio_x * match_ratio_y) / (match_ratio_x + match_ratio_y)
        return 2 * (match_ratio_x * match_ratio_y) / (match_ratio_x + match_ratio_y)

    def _get_noisy_items(
        self, nodes: t.List[Node], property_name: str, percent_cut_off: float
    ) -> t.List[str]:

        all_items = []
        for node in nodes:
            items = node.get_property(property_name)
            if items is not None:
                if isinstance(items, str):
                    all_items.append(items)
                elif isinstance(items, list):
                    all_items.extend(items)
                else:
                    pass

        num_unique_items = len(set(all_items))
        num_noisy_items = max(1, int(num_unique_items * percent_cut_off))
        noisy_list = list(dict(Counter(all_items).most_common()).keys())[
            :num_noisy_items
        ]
        return noisy_list

    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        if self.property_name is None:
            self.property_name

        distance_measure = self.distance_measure_map[self.distance_measure]
        noisy_items = self._get_noisy_items(
            kg.nodes, self.property_name, self.noise_threshold
        )
        relationships = []
        for i, node_x in enumerate(kg.nodes):
            if i % 500 == 0:
                print(f"Processing node {i} of {len(kg.nodes)}", flush=True)
            for j, node_y in enumerate(kg.nodes):
                if i >= j:
                    continue
                node_x_items = node_x.get_property(self.property_name)
                node_y_items = node_y.get_property(self.property_name)
                if node_x_items is None or node_y_items is None:
                    raise ValueError(
                        f"Node {node_x.id} or {node_y.id} has no {self.property_name}"
                    )
                if self.key_name is not None:
                    node_x_items = node_x_items.get(self.key_name, [])
                    node_y_items = node_y_items.get(self.key_name, [])

                # Filter out noisy items
                filtered_x_items = [x for x in node_x_items if x not in noisy_items]
                filtered_y_items = [y for y in node_y_items if y not in noisy_items]

                if not filtered_x_items or not filtered_y_items:
                    continue

                matched_x_indices = set()
                matched_y_indices = set()
                overlapped_items = []

                # Build a similarity matrix
                for x_idx, x in enumerate(filtered_x_items):
                    for y_idx, y in enumerate(filtered_y_items):
                        similarity = 1 - distance_measure.distance(x.lower(), y.lower())
                        if similarity >= self.distance_threshold:
                            matched_x_indices.add(x_idx)
                            matched_y_indices.add(y_idx)
                            overlapped_items.append((x, y))

                # Calculate the overlap score
                similarity = self._overlap_score(
                    len(matched_x_indices),
                    len(matched_y_indices),
                    len(node_x_items),
                    len(node_y_items),
                )

                is_valid_match = True
                if self.target_cross_source is not None:
                    is_valid_match = self.target_cross_source(
                        node_x
                    ) != self.target_cross_source(node_y)

                # We separate siblings into their own relationship type so we can put in place methods
                # to prevent search drift when querying the KG.
                nodes_are_not_siblings = _nodes_are_not_siblings(node_x, node_y)
                if nodes_are_not_siblings:
                    threshold = self.threshold
                else:
                    thresh_diff = 1 - self.threshold
                    thresh_diff = thresh_diff * 0.9
                    threshold = 1 - thresh_diff

                if similarity >= threshold and is_valid_match:
                    relationships.append(
                        Relationship(
                            source=node_x,
                            target=node_y,
                            bidirectional=True,
                            type=(
                                self.new_property_name
                                if nodes_are_not_siblings
                                else f"sibling_{self.new_property_name}"
                            ),
                            properties={
                                self.new_property_name: similarity,
                                "overlapped_items": overlapped_items,
                                "num_noisy_items": len(node_x_items)
                                + len(node_y_items)
                                - len(filtered_x_items)
                                - len(filtered_y_items),
                            },
                        )
                    )

        return relationships
