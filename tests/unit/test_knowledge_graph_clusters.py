import pytest
import uuid
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship


def create_document_node(name=None, id=None):
    """Helper function to create a document node with proper structure."""
    if id is None:
        id = uuid.uuid4()

    return Node(
        id=id,
        type=NodeType.DOCUMENT,
        properties={
            "page_content": f"{name or 'document'} content",
            "summary": f"{name or 'document'} summary",
            "document_metadata": {},
            "summary_embedding": [0.001, 0.002, 0.003],
            "themes": (
                [f"Theme {name}1", f"Theme {name}2"] if name else ["Theme 1", "Theme 2"]
            ),
            "entities": (
                [f"Entity {name}1", f"Entity {name}2"]
                if name
                else ["Entity 1", "Entity 2"]
            ),
        },
    )


def create_chunk_node(name=None, id=None):
    """Helper function to create a chunk node with proper structure."""
    if id is None:
        id = uuid.uuid4()

    return Node(
        id=id,
        type=NodeType.CHUNK,
        properties={
            "page_content": f"{name or 'chunk'} content",
            "summary": f"{name or 'chunk'} summary",
            "summary_embedding": [0.001, 0.002, 0.003],
            "themes": (
                [f"Theme {name}1", f"Theme {name}2"] if name else ["Theme 1", "Theme 2"]
            ),
            "entities": (
                [f"Entity {name}1", f"Entity {name}2"]
                if name
                else ["Entity 1", "Entity 2"]
            ),
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
        nodes.append(create_document_node(name=f"sim_node_{i+1}"))

    # Create chain of cosine similarity relationships
    relationships = []
    for i in range(node_count - 1):
        similarity_score = 0.8 + (i * 0.02)  # Vary the similarity score slightly
        rel = Relationship(
            source=nodes[i],
            target=nodes[i + 1],
            type="cosine_similarity",
            bidirectional=True,
            properties={"summary_similarity": similarity_score},
        )
        relationships.append(rel)

    return nodes, relationships


def create_chain_of_overlaps(starting_node, node_count=5):
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

    # Use starting_node as the first node
    nodes.append(starting_node)
    
    # Create remaining nodes, alternating between document and chunk
    for i in range(node_count - 1):
        if i % 2 == 0:
            nodes.append(create_document_node(name=f"overlap_node_{i+1}"))
        else:
            nodes.append(create_chunk_node(name=f"overlap_node_{i+1}"))

    # Create chain of entity overlap relationships
    relationships = []
    for i in range(node_count - 1):
        overlap_score = 0.3 + (i * 0.05)  # Vary the overlap score slightly

        # Create overlapped items based on node properties
        source_node = nodes[i]
        target_node = nodes[i + 1]

        source_entities = source_node.properties.get("entities", [])
        target_entities = target_node.properties.get("entities", [])

        # Create at least one overlapped item
        overlapped_items = []
        if source_entities and target_entities:
            overlapped_items.append([source_entities[0], target_entities[0]])

            # Add a second overlapped item if available
            if len(source_entities) > 1 and len(target_entities) > 1:
                overlapped_items.append([source_entities[1], target_entities[1]])

        rel = Relationship(
            source=source_node,
            target=target_node,
            type="entities_overlap",
            bidirectional=False,
            properties={
                "entities_overlap_score": overlap_score,
                "overlapped_items": overlapped_items,
            },
        )
        relationships.append(rel)

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
        "A": create_document_node(name="A"),
        "B": create_chunk_node(name="B"),
        "C": create_chunk_node(name="C"),
        "D": create_chunk_node(name="D"),
        "E": create_chunk_node(name="E"),
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
    nodes, relationships = create_chain_of_similarities(create_document_node(name="Start"), node_count=4)

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
    nodes, relationships = create_chain_of_overlaps(create_document_node(name="Start"), node_count=4)

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
    nodes, relationships = create_chain_of_similarities(create_document_node(name="Start"), node_count=3)

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
    nodes, relationships = create_chain_of_similarities(create_document_node(name="Start"), node_count=3)

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
    nodes, relationships = create_chain_of_overlaps(create_document_node(name="Start"), node_count=3)

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

    node_a = create_chunk_node(name="A", id=id_a)
    node_b = create_chunk_node(name="B", id=id_b)

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
    node_a = create_chunk_node(name="A")
    node_b = create_chunk_node(name="B")

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
