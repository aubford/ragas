import pytest
from typing import Any

import numpy as np
from langchain_core.callbacks import Callbacks

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node, Relationship
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.prompts import ConceptCombinations
from ragas.testset.synthesizers.prompts import PersonaThemesMapping
from tests.unit.test_knowledge_graph_clusters import (
    create_document_and_child_nodes,
    create_chain_of_similarities,
    build_knowledge_graph,
)


class MockConceptCombinationPrompt(PydanticPrompt):
    async def generate(self, data: Any, llm: Any, callbacks=None):
        # Return a predetermined ConceptCombinations object
        return ConceptCombinations(
            combinations=[
                ["Theme_A_1", "Theme_B_1"],
                ["Theme_A_2", "Theme_C_1"],
                ["Theme_B_2", "Theme_C_2"],
            ]
        )


class MockThemePersonaMatchingPrompt(PydanticPrompt):
    async def generate(self, data: Any, llm: Any, callbacks=None):
        # Return a predetermined PersonaThemesMapping object
        return PersonaThemesMapping(
            mapping={
                "Researcher": [
                    "Theme_A_1",
                    "Theme_A_2",
                    "Theme_B_1",
                    "Theme_B_2",
                    "Theme_C_1",
                    "Theme_C_2",
                ]
            }
        )


@pytest.mark.asyncio
async def test_generate_scenarios(fake_llm):
    """Test the _generate_scenarios method of MultiHopAbstractQuerySynthesizer."""
    nodes, relationships = create_document_and_child_nodes()
    sim_nodes, sim_relationships = create_chain_of_similarities(nodes[0], node_count=3)
    nodes.extend(sim_nodes)
    relationships.extend(sim_relationships)
    kg = build_knowledge_graph(nodes, relationships)

    # Create a list of personas
    personas = [
        Persona(
            name="Researcher",
            role_description="Researcher interested in the latest advancements in AI.",
        ),
        Persona(
            name="Engineer",
            role_description="Engineer interested in the latest advancements in AI.",
        ),
    ]

    # Create an instance of MultiHopAbstractQuerySynthesizer
    synthesizer = MultiHopAbstractQuerySynthesizer(llm=fake_llm)

    # Replace the prompts with mock versions
    synthesizer.concept_combination_prompt = MockConceptCombinationPrompt()
    synthesizer.theme_persona_matching_prompt = MockThemePersonaMatchingPrompt()
    
    n = 3

    # Call the _generate_scenarios method
    scenarios = await synthesizer._generate_scenarios(
        n=n,
        knowledge_graph=kg,
        persona_list=personas,
        callbacks=None,
    )

    # Assert that we got the expected number of scenarios
    assert len(scenarios) == n, f"Expected {n} scenarios, got {len(scenarios)}"

    # Check that each scenario has the expected properties
    for scenario in scenarios:
        assert hasattr(scenario, "nodes"), "Scenario should have nodes attribute"
        assert hasattr(scenario, "persona"), "Scenario should have persona attribute"
        assert hasattr(scenario, "style"), "Scenario should have style attribute"
        assert hasattr(scenario, "length"), "Scenario should have length attribute"
        assert hasattr(
            scenario, "combinations"
        ), "Scenario should have combinations attribute"

        # Check that the persona is from our list
        assert scenario.persona in personas, f"Unexpected persona: {scenario.persona}"

        # Check that the nodes are from our knowledge graph
        for node in scenario.nodes:
            assert node in nodes, f"Unexpected node: {node}"

        # Check that the combinations are from the themes we defined
        for item in scenario.combinations:
            assert item in [
                "Theme_A_1",
                "Theme_A_2",
                "Theme_B_1",
                "Theme_B_2",
                "Theme_C_1",
                "Theme_C_2",
            ], f"Unexpected item: {item}"
