import pytest
import uuid
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship


class DebugUUID(uuid.UUID):
    """
    A UUID subclass that displays a debug name instead of the UUID value.
    Creates a more readable graph representation in logs/debuggers while maintaining UUID compatibility.
    """

    def __init__(self, debug_name):
        # Create a random UUID internally
        self.debug = debug_name
        super().__init__(hex=str(uuid.uuid4()))

    def __str__(self):
        return self.debug

    def __repr__(self):
        return f"DebugUUID('{self.debug}')"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


def create_document_node(name):
    """Helper function to create a document node with proper structure."""
    return Node(
        id=DebugUUID(name),
        type=NodeType.DOCUMENT,
        properties={
            "page_content": f"{name} content",
            "summary": f"{name} summary",
            "document_metadata": {},
            "summary_embedding": [0.001, 0.002, 0.003],
            "themes": [],
            "entities": [],
        },
    )


def create_chunk_node(name):
    """Helper function to create a chunk node with proper structure."""
    return Node(
        id=DebugUUID(name),
        type=NodeType.CHUNK,
        properties={
            "page_content": f"{name} content",
            "summary": f"{name} summary",
            "summary_embedding": [0.001, 0.002, 0.003],
            "themes": [],
            "entities": [],
        },
    )


def create_chain_of_similarities(starting_node, node_count=5):
    """
    Create a chain of document nodes with cosine similarity relationships.

    Parameters
    ----------
    starting_node : Node
        Node to start the chain from. This will be the first node in the chain.
    node_count : int
        Number of nodes to create

    Returns
    -------
    tuple
        (list of nodes, list of relationships)
    """
    # Create document nodes
    nodes = []

    # Use starting_node as the first node
    nodes.append(starting_node)

    # Create remaining nodes
    for i in range(node_count - 1):
        nodes.append(create_document_node(name=f"sim_from_{starting_node.id}_{i + 1}"))

    # Create chain of cosine similarity relationships
    relationships = []
    for i in range(node_count - 1):
        rel = Relationship(
            source=nodes[i],
            target=nodes[i + 1],
            type="cosine_similarity",
            bidirectional=True,
            properties={"summary_similarity": 0.9},
        )
        relationships.append(rel)

    return nodes, relationships


def create_chain_of_overlap_nodes(starting_node, node_count=3):
    """
    Create a chain of nodes with entity overlap relationships.

    Parameters
    ----------
    starting_node : Node
        Node to start the chain from. This will be the first node in the chain.
    node_count : int
        Number of nodes to create

    Returns
    -------
    tuple
        (list of nodes, list of relationships)
    """
    # Create nodes (mix of document and chunk nodes)
    nodes = []
    relationships = []

    # Use starting_node as the first node and set its entity
    starting_node.properties["entities"] = [f"E_{starting_node.id}_1"]
    nodes.append(starting_node)

    # Create relationships and remaining node
    prev_node = starting_node
    for i in range(node_count - 1):
        # Realistic entity assignment
        prev_entity = f"E_{starting_node.id}_{i+1}"
        new_entity = f"E_{starting_node.id}_{i+2}"

        new_node = create_document_node(name=f"overlap_from_{starting_node.id}_{i + 1}")

        # Add entities to the new node, including overlap w/ previous node
        new_node.properties["entities"] = [prev_entity, new_entity]
        nodes.append(new_node)

        rel = Relationship(
            source=prev_node,
            target=new_node,
            type="entities_overlap",
            bidirectional=False,
            properties={
                "entities_overlap_score": 0.1,
                "overlapped_items": [[prev_entity, prev_entity]],
            },
        )
        relationships.append(rel)
        prev_node = new_node

    return nodes, relationships


def create_document_and_child_nodes():
    """
    Create a document node and its child chunk nodes with the same structure as create_branched_graph.

    Returns
    -------
    tuple
        (dict of nodes, list of relationships)
    """
    # Create nodes - A is a document, the rest are chunks
    nodes = {
        "A": create_document_node("A"),
        "B": create_chunk_node("B"),
        "C": create_chunk_node("C"),
        "D": create_chunk_node("D"),
        "E": create_chunk_node("E"),
    }

    # Create "child" relationships from document to chunks
    child_relationships = [
        Relationship(
            source=nodes["A"],
            target=nodes["B"],
            type="child",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["A"],
            target=nodes["C"],
            type="child",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["A"],
            target=nodes["D"],
            type="child",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["A"],
            target=nodes["E"],
            type="child",
            bidirectional=False,
            properties={},
        ),
    ]

    # Create "next" relationships between chunks
    next_relationships = [
        Relationship(
            source=nodes["B"],
            target=nodes["C"],
            type="next",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["C"],
            target=nodes["D"],
            type="next",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["D"],
            target=nodes["E"],
            type="next",
            bidirectional=False,
            properties={},
        ),
    ]

    # Combine all relationships
    relationships = child_relationships + next_relationships

    return nodes, relationships


def build_knowledge_graph(nodes, relationships):
    """
    Build a knowledge graph from nodes and relationships.

    Parameters
    ----------
    nodes : list or dict
        Nodes to add to the graph
    relationships : list
        Relationships to add to the graph

    Returns
    -------
    KnowledgeGraph
        The constructed knowledge graph
    """
    kg = KnowledgeGraph()

    # Add nodes to the graph
    if isinstance(nodes, dict):
        for node in nodes.values():
            kg.add(node)
    else:
        for node in nodes:
            kg.add(node)

    # Add relationships to the graph
    for rel in relationships:
        kg.add(rel)

    return kg


def test_find_indirect_clusters_with_document_and_children():
    """Test find_indirect_clusters with a document and its child nodes."""
    # Create nodes and relationships
    nodes, relationships = create_document_and_child_nodes()

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find clusters
    clusters = kg.find_indirect_clusters()

    # Verify clusters
    assert len(clusters) > 0

    # Check if we have clusters containing the expected paths
    has_abcd_path = any(
        nodes["A"] in cluster
        and nodes["B"] in cluster
        and nodes["C"] in cluster
        and nodes["D"] in cluster
        for cluster in clusters
    )
    has_abce_path = any(
        nodes["A"] in cluster
        and nodes["B"] in cluster
        and nodes["C"] in cluster
        and nodes["E"] in cluster
        for cluster in clusters
    )

    assert has_abcd_path
    assert has_abce_path


def test_find_indirect_clusters_with_similarity_relationships():
    """Test find_indirect_clusters with cosine similarity relationships between document nodes."""
    # Create nodes and relationships
    nodes, relationships = create_chain_of_similarities(
        create_document_node("Start"), node_count=4
    )

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find clusters
    clusters = kg.find_indirect_clusters()

    # Verify clusters
    assert len(clusters) > 0

    # Check if we have a cluster containing all nodes
    has_all_nodes = any(all(node in cluster for node in nodes) for cluster in clusters)

    assert has_all_nodes


def test_find_indirect_clusters_with_overlap_relationships():
    """Test find_indirect_clusters with entity overlap relationships."""
    # Create nodes and relationships
    nodes, relationships = create_chain_of_overlap_nodes(
        create_document_node("Start"), node_count=4
    )

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find clusters
    clusters = kg.find_indirect_clusters()

    # Verify clusters
    assert len(clusters) > 0

    # Check if we have a cluster containing all nodes
    has_all_nodes = any(all(node in cluster for node in nodes) for cluster in clusters)

    assert has_all_nodes


def test_find_indirect_clusters_with_condition():
    """Test find_indirect_clusters with a relationship condition."""
    # Create nodes and relationships
    nodes, relationships = create_document_and_child_nodes()

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find clusters with condition: only consider "next" relationships
    condition = lambda rel: rel.type == "next"
    clusters = kg.find_indirect_clusters(relationship_condition=condition)

    # Verify clusters
    assert len(clusters) > 0

    # Check if we have clusters with the expected paths
    # We should have B->C->D and C->E but not paths including A
    has_bcd_path = any(
        nodes["B"] in cluster
        and nodes["C"] in cluster
        and nodes["D"] in cluster
        and nodes["A"] not in cluster
        for cluster in clusters
    )
    has_ce_path = any(
        nodes["C"] in cluster and nodes["E"] in cluster and nodes["A"] not in cluster
        for cluster in clusters
    )

    assert has_bcd_path
    assert has_ce_path


def test_find_indirect_clusters_with_bidirectional():
    """Test find_indirect_clusters with bidirectional relationships."""
    # Create document nodes with similarity relationships (which are bidirectional)
    nodes, relationships = create_chain_of_similarities(
        create_document_node("Start"), node_count=3
    )

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find clusters
    clusters = kg.find_indirect_clusters()

    # Verify clusters
    assert len(clusters) > 0

    # Check if we have a cluster containing all nodes
    has_all_nodes = any(all(node in cluster for node in nodes) for cluster in clusters)

    assert has_all_nodes


def test_find_indirect_clusters_depth_limit():
    """Test find_indirect_clusters with a depth limit."""
    # Create nodes and relationships
    nodes, relationships = create_document_and_child_nodes()

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find clusters with depth limit 1
    clusters_depth_1 = kg.find_indirect_clusters(depth_limit=1)

    # Verify clusters with depth limit 1
    # We should have pairs but not longer paths
    assert len(clusters_depth_1) > 0

    # No cluster should contain all five nodes
    assert not any(
        all(node in cluster for node in nodes.values()) for cluster in clusters_depth_1
    )

    # Find clusters with depth limit 3 (default)
    clusters_depth_3 = kg.find_indirect_clusters()

    # Verify clusters with depth limit 3
    # We should have the full paths
    assert any(
        nodes["A"] in cluster
        and nodes["B"] in cluster
        and nodes["C"] in cluster
        and nodes["D"] in cluster
        for cluster in clusters_depth_3
    )


def test_find_two_nodes_single_rel_with_document_and_children():
    """Test find_two_nodes_single_rel with document and child nodes."""
    # Create nodes and relationships
    nodes, relationships = create_document_and_child_nodes()

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find two-node relationships
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets
    assert len(triplets) == 6  # 3 child relationships + 3 next relationships

    # Check if we have the expected triplets
    child_triplets = [t for t in triplets if t[1].type == "child"]
    next_triplets = [t for t in triplets if t[1].type == "next"]

    assert len(child_triplets) == 3
    assert len(next_triplets) == 3

    # Check child relationships
    has_ab_child = any(
        triplet[0] == nodes["A"]
        and triplet[2] == nodes["B"]
        and triplet[1].type == "child"
        for triplet in triplets
    )
    has_ac_child = any(
        triplet[0] == nodes["A"]
        and triplet[2] == nodes["C"]
        and triplet[1].type == "child"
        for triplet in triplets
    )
    has_ad_child = any(
        triplet[0] == nodes["A"]
        and triplet[2] == nodes["D"]
        and triplet[1].type == "child"
        for triplet in triplets
    )

    assert has_ab_child
    assert has_ac_child
    assert has_ad_child

    # Check next relationships
    has_bc_next = any(
        triplet[0] == nodes["B"]
        and triplet[2] == nodes["C"]
        and triplet[1].type == "next"
        for triplet in triplets
    )
    has_cd_next = any(
        triplet[0] == nodes["C"]
        and triplet[2] == nodes["D"]
        and triplet[1].type == "next"
        for triplet in triplets
    )
    has_ce_next = any(
        triplet[0] == nodes["C"]
        and triplet[2] == nodes["E"]
        and triplet[1].type == "next"
        for triplet in triplets
    )

    assert has_bc_next
    assert has_cd_next
    assert has_ce_next


def test_find_two_nodes_single_rel_with_similarity():
    """Test find_two_nodes_single_rel with cosine similarity relationships."""
    # Create nodes and relationships
    nodes, relationships = create_chain_of_similarities(
        create_document_node("Start"), node_count=3
    )

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find two-node relationships
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets
    assert len(triplets) == 2

    # Check if all triplets have the correct relationship type
    for triplet in triplets:
        assert triplet[1].type == "cosine_similarity"
        assert "summary_similarity" in triplet[1].properties


def test_find_two_nodes_single_rel_with_overlap():
    """Test find_two_nodes_single_rel with entity overlap relationships."""
    # Create nodes and relationships
    nodes, relationships = create_chain_of_overlap_nodes(
        create_document_node("Start"), node_count=3
    )

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find two-node relationships
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets
    assert len(triplets) == 2

    # Check if all triplets have the correct relationship type
    for triplet in triplets:
        assert triplet[1].type == "entities_overlap"
        assert "entities_overlap_score" in triplet[1].properties
        assert "overlapped_items" in triplet[1].properties


def test_find_two_nodes_single_rel_with_condition():
    """Test find_two_nodes_single_rel with a relationship condition."""
    # Create nodes and relationships
    nodes, relationships = create_document_and_child_nodes()

    # Build knowledge graph
    kg = build_knowledge_graph(nodes, relationships)

    # Find two-node relationships with condition: only consider "child" relationships
    condition = lambda rel: rel.type == "child"
    triplets = kg.find_two_nodes_single_rel(relationship_condition=condition)

    # Verify triplets
    assert len(triplets) == 3  # Should have 3 child relationships

    # Check if all triplets have the correct relationship type
    for triplet in triplets:
        assert triplet[1].type == "child"
        assert triplet[0] == nodes["A"]  # Source should be the document node


def test_find_two_nodes_single_rel_normalized_order():
    """Test that find_two_nodes_single_rel normalizes the order of nodes based on ID."""
    # Create nodes with specific UUIDs to ensure consistent ordering
    id_a = uuid.UUID("00000000-0000-0000-0000-000000000001")
    id_b = uuid.UUID("00000000-0000-0000-0000-000000000002")

    node_a = create_chunk_node("chunk_A")
    node_b = create_chunk_node("chunk_B")

    # Create relationship from B to A (reverse of ID order)
    rel_ba = Relationship(
        source=node_b, target=node_a, type="next", bidirectional=False, properties={}
    )

    # Build knowledge graph
    kg = build_knowledge_graph([node_a, node_b], [rel_ba])

    # Find two-node relationships
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets - should be normalized with node_a first due to smaller ID
    assert len(triplets) == 1
    assert triplets[0][0] == node_a  # Smaller ID should be first
    assert triplets[0][2] == node_b  # Larger ID should be second


def test_find_two_nodes_single_rel_with_self_loops():
    """Test find_two_nodes_single_rel with self-loops (should be excluded)."""
    # Create nodes
    node_a = create_chunk_node("A")
    node_b = create_chunk_node("B")

    # Create relationships including a self-loop
    rel_ab = Relationship(
        source=node_a, target=node_b, type="next", bidirectional=False, properties={}
    )
    rel_aa = Relationship(
        source=node_a,
        target=node_a,
        type="self_loop",
        bidirectional=True,
        properties={},
    )

    # Build knowledge graph
    kg = build_knowledge_graph([node_a, node_b], [rel_ab, rel_aa])

    # Find two-node relationships
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets - self-loops should be excluded
    assert len(triplets) == 1

    # Check if we have only the A-B relationship
    assert triplets[0][0] == node_a
    assert triplets[0][2] == node_b
    assert triplets[0][1].type == "next"
