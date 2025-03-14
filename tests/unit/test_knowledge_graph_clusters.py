import pytest
import uuid
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
import random
import time


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


def create_chain_of_similarities(starting_node, node_count=5, cycle=False):
    """
    Create a chain of document nodes with cosine similarity relationships.

    Parameters
    ----------
    starting_node : Node
        Node to start the chain from. This will be the first node in the chain.
    node_count : int
        Number of nodes to create
    cycle : bool
        If True, add a relationship from the last node back to the first node

    Returns
    -------
    tuple
        (list of nodes, list of relationships)
    """
    # Use starting_node as the first node
    nodes = [starting_node]

    # Create remaining nodes
    for i in range(node_count - 1):
        nodes.append(create_document_node(name=f"{starting_node.id}_{i + 1}"))

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

    if cycle and node_count > 1:
        # For the cycle, the last node should share an entity with the first node
        cycle_rel = Relationship(
            source=nodes[-1],
            target=nodes[0],
            type="cosine_similarity",
            bidirectional=True,
            properties={"summary_similarity": 0.9},
        )
        relationships.append(cycle_rel)

    return nodes, relationships


def create_chain_of_overlaps(starting_node, node_count=3, cycle=False):
    """
    Create a chain of nodes with entity overlap relationships.

    Parameters
    ----------
    starting_node : Node
        Node to start the chain from. This will be the first node in the chain.
    node_count : int
        Number of nodes to create
    cycle : bool
        If True, add a relationship from the last node back to the first node

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
        prev_entity = f"E_{starting_node.id}_{i + 1}"
        new_entity = f"E_{starting_node.id}_{i + 2}"

        new_node = create_document_node(name=f"{starting_node.id}_{i + 1}")

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

    if cycle and node_count > 1:
        # For the cycle, the last node should share an entity with the first node
        first_entity = f"E_{starting_node.id}_1"
        nodes[-1].properties["entities"].append(first_entity)

        cycle_rel = Relationship(
            source=nodes[-1],
            target=nodes[0],
            type="entities_overlap",
            bidirectional=False,
            properties={
                "entities_overlap_score": 0.1,
                "overlapped_items": [[first_entity, first_entity]],
            },
        )
        relationships.append(cycle_rel)

    return nodes, relationships


def create_web_of_similarities(node_count=4, similarity_score=0.9):
    """
    Create a web of document nodes with cosine similarity relationships between them.
    This represents the worst case scenario knowledge graph for the node_count in terms
    of time complexity.

    Parameters
    ----------
    node_count : int
        Number of nodes to create
    similarity_score : float
        Similarity score to use for all relationships

    Returns
    -------
    tuple
        (list of nodes, list of relationships)
    """
    # Create nodes
    nodes = []
    for i in range(node_count):
        nodes.append(create_document_node(name=f"{i}"))

    # Create relationships
    relationships = []
    for i in range(node_count):
        for j in range(node_count):
            if i != j:  # Don't connect node to itself
                rel = Relationship(
                    source=nodes[i],
                    target=nodes[j],
                    type="cosine_similarity",
                    bidirectional=True,
                    properties={"summary_similarity": similarity_score},
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


def assert_clusters_equal(actual_clusters, expected_clusters):
    """
    Helper function to compare clusters with unordered comparison.

    Args:
        actual_clusters: List of sets representing the actual clusters
        expected_clusters: List of sets representing the expected clusters
    """
    # Convert both lists to sets of frozensets for unordered comparison
    actual_clusters_set = {frozenset(cluster) for cluster in actual_clusters}
    expected_clusters_set = {frozenset(cluster) for cluster in expected_clusters}

    assert (
        actual_clusters_set == expected_clusters_set
    ), f"Expected clusters: {expected_clusters_set}\nActual clusters: {actual_clusters_set}"


def assert_n_clusters_with_varying_params(kg, param_list):
    """
    Helper function to test find_n_indirect_clusters with various combinations of n and depth_limit.

    Args:
        kg: KnowledgeGraph instance to test
        param_list: List of tuples (n, depth_limit) to test
    """
    for n, depth_limit in param_list:
        clusters = kg.find_n_indirect_clusters(n=n, depth_limit=depth_limit)
        if len(clusters) != n:
            # Convert clusters to sets of node IDs for more readable error messages
            cluster_ids = [{str(node.id) for node in cluster} for cluster in clusters]
            pytest.fail(
                f"Expected {n} clusters with params (n={n}, depth_limit={depth_limit}), "
                f"but got {len(clusters)} clusters.\n"
                f"Actual clusters: {cluster_ids}"
            )


def test_find_indirect_clusters_with_document_and_children():
    """Test find_indirect_clusters for a document (A) and its child nodes (B, C, D, E)."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=4)

    assert_clusters_equal(
        clusters,
        [
            {nodes["A"], nodes["B"]},
            {nodes["A"], nodes["C"]},
            {nodes["A"], nodes["D"]},
            {nodes["A"], nodes["E"]},
            {nodes["B"], nodes["C"]},
            {nodes["C"], nodes["D"]},
            {nodes["D"], nodes["E"]},
            {nodes["A"], nodes["B"], nodes["C"]},
            {nodes["A"], nodes["C"], nodes["D"]},
            {nodes["A"], nodes["D"], nodes["E"]},
            {nodes["B"], nodes["C"], nodes["D"]},
            {nodes["C"], nodes["D"], nodes["E"]},
            {nodes["A"], nodes["B"], nodes["C"], nodes["D"]},
            {nodes["A"], nodes["C"], nodes["D"], nodes["E"]},
            {nodes["B"], nodes["C"], nodes["D"], nodes["E"]},
        ],
    )


def test_find_n_indirect_clusters_with_document_and_children():
    """Test find_n_indirect_clusters for a document (A) and its child nodes (B, C, D, E)."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)

    # It should not include subsets of found nodes
    clusters = kg.find_n_indirect_clusters(n=4, depth_limit=4)
    assert_clusters_equal(
        clusters,
        [
            {nodes["A"], nodes["B"], nodes["C"], nodes["D"]},
            {nodes["A"], nodes["C"], nodes["D"], nodes["E"]},
            {nodes["B"], nodes["C"], nodes["D"], nodes["E"]},
        ],
    )

    # Test different combinations of n and depth_limit parameters
    assert_n_clusters_with_varying_params(
        kg, [(3, 3), (3, 2), (2, 4), (2, 3), (2, 2), (1, 2)]
    )


def test_find_indirect_clusters_with_similarity_relationships():
    """Test find_indirect_clusters with cosine similarity relationships between document nodes."""
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=4
    )
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=4)

    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1]},
            {nodes[1], nodes[2]},
            {nodes[2], nodes[3]},
            {nodes[0], nodes[1], nodes[2]},
            {nodes[1], nodes[2], nodes[3]},
            {nodes[0], nodes[1], nodes[2], nodes[3]},
        ],
    )


def test_find_n_indirect_clusters_with_similarity_relationships():
    """
    Test find_n_indirect_clusters with bidirectional cosine similarity relationships between document nodes.
    Test that we handle cycles and branches correctly.
    """
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=4
    )
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=5, depth_limit=4)

    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1], nodes[2], nodes[3]},
        ],
    )

    # create 5 node cycle branching off node 2
    five_node_cycle, fnc_relationships = create_chain_of_similarities(
        nodes[2], node_count=5, cycle=True
    )
    # create independent 2 node cycle to cover edge case
    two_node_cycle, tnc_relationships = create_chain_of_similarities(
        create_document_node("C"), node_count=2, cycle=True
    )

    new_nodes = five_node_cycle[1:] + two_node_cycle
    nodes.extend(new_nodes)
    for item in new_nodes + fnc_relationships + tnc_relationships:
        kg.add(item)

    clusters = kg.find_n_indirect_clusters(n=12, depth_limit=3)

    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1], nodes[2]},
            {nodes[1], nodes[2], nodes[3]},
            {nodes[2], nodes[3], nodes[4]},
            {nodes[1], nodes[2], nodes[4]},
            {nodes[1], nodes[2], nodes[7]},
            {nodes[2], nodes[4], nodes[5]},
            {nodes[2], nodes[4], nodes[7]},
            {nodes[2], nodes[3], nodes[7]},
            {nodes[2], nodes[6], nodes[7]},
            {nodes[4], nodes[5], nodes[6]},
            {nodes[5], nodes[6], nodes[7]},
            {nodes[8], nodes[9]},  # independent two node cycle
        ],
    )

    # Test different combinations of n and depth_limit parameters
    assert_n_clusters_with_varying_params(
        kg, [(4, 4), (4, 3), (4, 2), (3, 4), (3, 3), (3, 2), (2, 4), (2, 3), (2, 2)]
    )


def test_find_indirect_clusters_with_overlap_relationships():
    """Test find_indirect_clusters with directional entity overlap relationships."""
    nodes, relationships = create_chain_of_overlaps(
        create_document_node("A"), node_count=4
    )
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=3)

    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1]},
            {nodes[1], nodes[2]},
            {nodes[2], nodes[3]},
            {nodes[0], nodes[1], nodes[2]},
            {nodes[1], nodes[2], nodes[3]},
        ],
    )


def test_find_n_indirect_clusters_with_overlap_relationships():
    """
    Test find_n_indirect_clusters with directional entity overlap relationships.
    Test that we handle cycles and branches correctly.
    """
    nodes, relationships = create_chain_of_overlaps(
        create_document_node("A"), node_count=4
    )
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=5, depth_limit=3)

    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1], nodes[2]},
            {nodes[1], nodes[2], nodes[3]},
        ],
    )

    # create 5 node cycle branching off node 2
    five_node_cycle, fnc_relationships = create_chain_of_overlaps(
        nodes[2], node_count=5, cycle=True
    )
    # create independent 2 node cycle to cover edge case
    two_node_cycle, tnc_relationships = create_chain_of_overlaps(
        create_document_node("C"), node_count=2, cycle=True
    )

    new_nodes = five_node_cycle[1:] + two_node_cycle
    nodes.extend(new_nodes)
    for item in new_nodes + fnc_relationships + tnc_relationships:
        kg.add(item)

    clusters = kg.find_n_indirect_clusters(n=15, depth_limit=3)

    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1], nodes[2]},
            {nodes[1], nodes[2], nodes[3]},
            {nodes[1], nodes[2], nodes[4]},
            {nodes[2], nodes[4], nodes[5]},
            {nodes[4], nodes[5], nodes[6]},
            {nodes[5], nodes[6], nodes[7]},
            {nodes[6], nodes[7], nodes[2]},
            {nodes[7], nodes[2], nodes[3]},
            {nodes[7], nodes[2], nodes[4]},
            {nodes[8], nodes[9]},  # independent two node cycle
        ],
    )

    # Test different combinations of n and depth_limit parameters
    assert_n_clusters_with_varying_params(
        kg, [(3, 4), (3, 4), (3, 3), (3, 2), (2, 4), (2, 3), (2, 2)]
    )


def test_find_n_indirect_clusters_handles_worst_case_grouping():
    """
    Test that the algorithm will always return n indirect clusters when all nodes are grouped into independent clusters
    of `n` nodes. This is a worst-case scenario that can lead to significant under-sampling if not handled correctly.
    """
    # The edge case is dependent on random.shuffle() so set a specific seed that exposes it deterministically.
    # Otherwise it only fails 50% of the time (when the 2 starting nodes are from the same cluster).
    original_state = random.getstate()
    random.seed(5)

    try:
        nodes_A, relationships_A = create_chain_of_similarities(
            create_document_node("A"), node_count=2
        )
        nodes_B, relationships_B = create_chain_of_similarities(
            create_document_node("B"), node_count=2
        )
        kg = build_knowledge_graph(nodes_A + nodes_B, relationships_A + relationships_B)
        clusters = kg.find_n_indirect_clusters(n=2, depth_limit=2)

        assert_clusters_equal(
            clusters,
            [
                {nodes_A[0], nodes_A[1]},
                {nodes_B[0], nodes_B[1]},
            ],
        )
    finally:
        # Restore original random state to avoid affecting other tests
        random.setstate(original_state)


def test_find_indirect_clusters_with_condition():
    """Test find_indirect_clusters with a relationship condition."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)

    def condition(rel):
        return rel.type == "next"

    clusters = kg.find_indirect_clusters(relationship_condition=condition)

    # Only "next" relationships are considered, so we should only have paths between B, C, D, and E
    assert_clusters_equal(
        clusters,
        [
            {nodes["B"], nodes["C"]},
            {nodes["C"], nodes["D"]},
            {nodes["D"], nodes["E"]},
            {nodes["B"], nodes["C"], nodes["D"]},
            {nodes["C"], nodes["D"], nodes["E"]},
        ],
    )


def test_find_n_indirect_clusters_with_condition():
    """Test find_n_indirect_clusters with a relationship condition."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)

    def condition(rel):
        return rel.type == "next"

    clusters = kg.find_n_indirect_clusters(n=5, relationship_condition=condition)

    # Only "next" relationships are considered, so we should only have paths between B, C, D, and E
    assert_clusters_equal(
        clusters,
        [
            {nodes["B"], nodes["C"], nodes["D"]},
            {nodes["C"], nodes["D"], nodes["E"]},
        ],
    )

    assert_n_clusters_with_varying_params(kg, [(2, 3), (2, 2)])


# test cyclic relationships for bidirectional relationships
def test_find_indirect_clusters_with_cyclic_similarity_relationships():
    """Test find_indirect_clusters with cyclic cosine similarity relationships."""
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=3, cycle=True
    )
    # branch off last node so it both cycles and branches
    branched_nodes, branched_relationships = create_chain_of_similarities(
        nodes[-1], node_count=2
    )
    nodes.extend(branched_nodes[1:])
    relationships.extend(branched_relationships)

    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=10)

    # With a cycle, we expect additional clusters that include paths through the cycle
    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1]},
            {nodes[1], nodes[2]},
            {nodes[2], nodes[3]},
            {nodes[2], nodes[0]},
            {nodes[0], nodes[1], nodes[2]},
            {nodes[0], nodes[2], nodes[3]},
            {nodes[1], nodes[2], nodes[0]},
            {nodes[2], nodes[0], nodes[1]},
            {nodes[1], nodes[2], nodes[3]},
            {nodes[0], nodes[1], nodes[2], nodes[3]},
        ],
    )


# test cyclic relationships for bidirectional relationships
def test_find_n_indirect_clusters_with_cyclic_similarity_relationships():
    """Test find_n_indirect_clusters with cyclic cosine similarity relationships."""
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=3, cycle=True
    )
    # branch off last node so it both cycles and branches
    branched_nodes, branched_relationships = create_chain_of_similarities(
        nodes[-1], node_count=2
    )
    nodes.extend(branched_nodes[1:])
    relationships.extend(branched_relationships)

    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=5, depth_limit=3)

    # With a cycle, we expect additional clusters that include paths through the cycle
    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1], nodes[2]},
            {nodes[0], nodes[2], nodes[3]},
            {nodes[1], nodes[2], nodes[0]},
            {nodes[2], nodes[0], nodes[1]},
            {nodes[1], nodes[2], nodes[3]},
        ],
    )

    assert_n_clusters_with_varying_params(kg, [(1, 4), (3, 3), (2, 3), (2, 2)])


def test_find_indirect_clusters_with_web_graph():
    """Test find_indirect_clusters with a spider web graph where all nodes connect to all other nodes."""
    nodes, relationships = create_web_of_similarities(node_count=4)

    # Convert nodes list to dictionary for easier assertion
    node_dict = {f"{i}": nodes[i] for i in range(len(nodes))}

    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=3)

    assert_clusters_equal(
        clusters,
        [
            {node_dict["0"], node_dict["1"]},
            {node_dict["0"], node_dict["2"]},
            {node_dict["0"], node_dict["3"]},
            {node_dict["1"], node_dict["2"]},
            {node_dict["1"], node_dict["3"]},
            {node_dict["2"], node_dict["3"]},
            {node_dict["0"], node_dict["1"], node_dict["2"]},
            {node_dict["0"], node_dict["1"], node_dict["3"]},
            {node_dict["0"], node_dict["2"], node_dict["3"]},
            {node_dict["1"], node_dict["2"], node_dict["3"]},
        ],
    )


def test_find_n_indirect_clusters_with_web_graph():
    """Test find_n_indirect_clusters with a spider web graph where all nodes connect to all other nodes."""
    nodes, relationships = create_web_of_similarities(node_count=4)

    # Convert nodes list to dictionary for easier assertion
    node_dict = {f"{i}": nodes[i] for i in range(len(nodes))}

    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=10, depth_limit=3)

    assert_clusters_equal(
        clusters,
        [
            {node_dict["0"], node_dict["1"], node_dict["2"]},
            {node_dict["0"], node_dict["1"], node_dict["3"]},
            {node_dict["0"], node_dict["2"], node_dict["3"]},
            {node_dict["1"], node_dict["2"], node_dict["3"]},
        ],
    )

    assert_n_clusters_with_varying_params(
        kg, [(4, 3), (3, 3), (3, 2), (2, 3), (2, 2), (1, 2)]
    )


def test_performance_find_n_indirect_clusters_max_density():
    """
    Test the time complexity performance of find_n_indirect_clusters with graphs of maximum density.
    Capping sampling relative to n should keep the time complexity <cubic.
    """
    # List of graph sizes to test (number of nodes)
    graph_sizes = [5, 10, 20, 80]
    results = []

    for size in graph_sizes:
        nodes, relationships = create_web_of_similarities(node_count=size)
        kg = build_knowledge_graph(nodes, relationships)

        # Measure execution time
        start_time = time.time()
        clusters = kg.find_n_indirect_clusters(n=size, depth_limit=4)
        end_time = time.time()

        execution_time = end_time - start_time

        # Store results
        results.append(
            {"size": size, "time": execution_time, "clusters": len(clusters)}
        )

        # Make sure we actually got the clusters
        assert len(clusters) == size

    # Print all results at the end of the test
    print("\nPerformance test results:")
    print("------------------------")
    print("Size | Time (s)")
    print("------------------------")

    for result in results:
        print(f"{result['size']:4d} | {result['time']:.6f}")

    print("------------------------")

    # Check if time complexity is reasonable
    for i in range(1, len(results)):
        size_ratio = results[i]["size"] / results[i - 1]["size"]
        time_ratio = results[i]["time"] / results[i - 1]["time"]
        # Goal is better than cubic since relationships grow exponentially with n and graph_size for a worst-case "web" graph.
        scaled_size_ratio = size_ratio**3
        print(
            f"Size ratio: {size_ratio:.2f}, Time ratio: {time_ratio:.2f}, Scaled ratio: {scaled_size_ratio:.2f}"
        )

        assert (
            time_ratio < scaled_size_ratio
        ), f"Time complexity growing faster than expected: size {results[i]['size']} vs {results[i-1]['size']}, time ratio {time_ratio:.2f} vs {scaled_size_ratio:.2f}"


@pytest.fixture
def constant_n_knowledge_graphs():
    """Returns the three knowledge graphs of increasing size."""
    graph_sizes = [10, 50, 500]
    knowledge_graphs = []

    for size in graph_sizes:
        nodes, relationships = create_web_of_similarities(node_count=size)
        kg = build_knowledge_graph(nodes, relationships)
        knowledge_graphs.append((kg, size))

    return knowledge_graphs


def test_performance_find_n_indirect_clusters_large_web_constant_n(
    constant_n_knowledge_graphs,
):
    """
    Test the time complexity performance of find_n_indirect_clusters with a constant n=10
    but dramatically increasing graph sizes. This tests how the algorithm scales when we're
    only interested in sampling a fixed number of clusters but may have a big graph.
    """
    constant_n = 10
    results = []

    for kg, size in constant_n_knowledge_graphs:
        # Measure execution time
        start_time = time.time()
        clusters = kg.find_n_indirect_clusters(n=constant_n, depth_limit=3)
        end_time = time.time()

        execution_time = end_time - start_time

        # Store results
        results.append(
            {
                "size": size,
                "n": constant_n,
                "time": execution_time,
                "clusters": len(clusters),
            }
        )

        # Make sure we got clusters (may be less than n if graph doesn't support that many)
        assert (
            len(clusters) <= constant_n
        ), f"Expected at most {constant_n} clusters, got {len(clusters)}"

    # Print all results at the end of the test
    print("\nPerformance test results (constant n=10):")
    print("----------------------------------")
    print("Graph Size | n | Clusters | Time (s)")
    print("----------------------------------")

    for result in results:
        print(
            f"{result['size']:10d} | {result['n']:1d} | {result['clusters']:8d} | {result['time']:.6f}"
        )

    print("----------------------------------")

    # Check if time complexity is reasonable
    for i in range(1, len(results)):
        size_ratio = results[i]["size"] / results[i - 1]["size"]
        time_ratio = results[i]["time"] / results[i - 1]["time"]

        scaled_size_ratio = size_ratio**2.5
        print(
            f"Size ratio: {size_ratio:.2f}, (Scaled: {scaled_size_ratio:.2f}), Time ratio: {time_ratio:.2f}"
        )

        assert (
            time_ratio < scaled_size_ratio
        ), f"Time complexity growing faster than expected: size {results[i]['size']} vs {results[i - 1]['size']}, time ratio {time_ratio:.2f} vs {scaled_size_ratio:.2f}"


def test_performance_find_n_indirect_clusters_independent_chains():
    """
    Test the time complexity performance of find_n_indirect_clusters with independent chains of 4 nodes.
    This uses the inflated sample size that is triggered when the nodes are isolated such that edges are less than nodes.
    """
    # List of total node counts to test
    graph_sizes = [8, 16, 32, 128, 1024]
    results = []

    for size in graph_sizes:
        # Calculate how many chains of 4 nodes we need
        num_chains = size // 4

        # Create independent chains of 4 nodes each
        all_nodes = []
        all_relationships = []

        for i in range(num_chains):
            chain_nodes, chain_relationships = create_chain_of_similarities(
                create_document_node(f"{i}_start"), node_count=4, cycle=False
            )
            all_nodes.extend(chain_nodes)
            all_relationships.extend(chain_relationships)

        kg = build_knowledge_graph(all_nodes, all_relationships)

        # Measure execution time
        start_time = time.time()
        clusters = kg.find_n_indirect_clusters(n=num_chains, depth_limit=3)
        end_time = time.time()

        execution_time = end_time - start_time

        # Store results
        results.append(
            {
                "size": size,
                "chains": num_chains,
                "time": execution_time,
                "clusters": len(clusters),
            }
        )

        # Make sure we got the expected number of clusters (one per chain)
        assert (
            len(clusters) == num_chains
        ), f"Expected {num_chains} clusters, got {len(clusters)}"

    # Print all results at the end of the test
    print("\nPerformance test results (independent chains):")
    print("------------------------")
    print("Size | Chains | Time (s)")
    print("------------------------")

    for result in results:
        print(f"{result['size']:4d} | {result['chains']:6d} | {result['time']:.6f}")

    print("------------------------")

    for i in range(1, len(results)):
        size_ratio = results[i]["size"] / results[i - 1]["size"]
        time_ratio = results[i]["time"] / results[i - 1]["time"]
        # Goal is to be ~quadratic or better.
        scaled_size_ratio = size_ratio**2
        print(
            f"Size ratio: {size_ratio:.2f} (scaled: {scaled_size_ratio:.2f}), Time ratio: {time_ratio:.2f}"
        )

        assert (
            time_ratio < scaled_size_ratio
        ), f"Time complexity growing faster than expected: size {results[i]['size']} vs {results[i-1]['size']}, time ratio {time_ratio:.2f} vs {scaled_size_ratio:.2f}"
