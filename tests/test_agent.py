from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import starter.agent as agent_module
from local_embeddings import _compatible_model_names
from starter.agent import Agent, SearchPlan


PRODUCTS = [
    {
        "parent_asin": "BELT1",
        "title": "Full Grain Leather Belt",
        "features": ["Imported", "Pull On closure", "100% Leather"],
        "description": ["Everyday men's belt"],
        "price": 35.0,
        "categories": ["Men", "Accessories", "Belts"],
        "details": {"Material": "Leather"},
        "average_rating": 4.7,
        "rating_number": 100,
        "store": "Example",
    },
    {
        "parent_asin": "BELT2",
        "title": "Canvas Web Belt",
        "features": ["Adjustable buckle"],
        "description": ["Casual canvas belt"],
        "price": 20.0,
        "categories": ["Men", "Accessories", "Belts"],
        "details": {"Material": "Canvas"},
        "average_rating": 4.5,
        "rating_number": 80,
        "store": "Example",
    },
    {
        "parent_asin": "SHOE1",
        "title": "Water Resistant Walking Shoe",
        "features": ["Lightweight", "Rubber sole"],
        "description": ["Comfortable travel walking shoe"],
        "price": 70.0,
        "categories": ["Women", "Shoes", "Walking"],
        "details": {"Color": "Blue"},
        "average_rating": 4.8,
        "rating_number": 200,
        "store": "Example",
    },
]


class FakeEmbeddingIndex:
    def search(self, text: str, top_k: int) -> list[tuple[str, float]]:
        return [("SHOE1", 0.91)]


class AgentPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_dir.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )
        with patch.object(agent_module, "LocalEmbeddingIndex", None):
            self.agent = Agent(self.catalog_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_multi_value_answer_is_remembered_as_separate_facts(self) -> None:
        self.agent.reset("session", {})
        first = self.agent.respond(
            "session", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10
        )
        self.assertEqual(first["ask_attribute"], "feature")
        second = self.agent.respond(
            "session", "For that, what matters is: Imported; Pull On closure.", 2, 10
        )
        facts = [fact.value for fact in self.agent._states["session"].facts]
        self.assertIn("Imported", facts)
        self.assertIn("Pull On closure", facts)
        self.assertTrue(second["recommendations"])

    def test_override_removes_soft_fact_but_preserves_hard_fact(self) -> None:
        self.agent.reset("session", {})
        state = self.agent._states["session"]
        state.add_fact("category", "Shirts Polos", 1)
        state.add_fact("material", "60% cotton 40% polyester", 2)
        state.add_fact("feature", "Button closure", 2)
        self.agent._update_state(
            state, "Actually, ignore my earlier preference. What I need is: cotton.", 3
        )
        values = [fact.value for fact in state.facts]
        self.assertIn("Shirts Polos", values)
        self.assertIn("60% cotton 40% polyester", values)
        self.assertNotIn("Button closure", values)

    def test_dense_route_joins_lexical_candidate_union(self) -> None:
        self.agent.embedding_index = FakeEmbeddingIndex()
        self.agent.reset("session", {})
        state = self.agent._states["session"]
        self.agent._update_state(state, "I'm looking for Accessories Belts.", 1)
        result = self.agent._retrieve(self.agent._fallback_plan(state), state, 10)
        self.assertIn("dense", result.route_ranks)
        self.assertIn("SHOE1", result.scores)

    def test_explicit_exclusion_is_carried_into_search_plan(self) -> None:
        self.agent.reset("session", {})
        state = self.agent._states["session"]
        self.agent._update_state(
            state, "I'm looking for Shoes Walking, but avoid leather.", 1
        )
        plan = self.agent._fallback_plan(state)
        self.assertEqual(plan.excluded_terms, ["leather"])

    def test_response_contract_returns_recommendations_while_asking(self) -> None:
        self.agent.reset("session", {"preference_tags": ["comfort"]})
        response = self.agent.respond(
            "session", "I'm looking for Shoes Walking, but I'm still exploring.", 1, 10
        )
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], agent_module.ALLOWED_ASK_ATTRIBUTES)
        self.assertGreater(len(response["recommendations"]), 0)
        self.assertLessEqual(len(response["recommendations"]), 10)

    def test_llm_plan_is_validated(self) -> None:
        self.agent.reset("session", {})
        state = self.agent._states["session"]
        self.agent._update_state(state, "I need blue walking shoes under $80.", 1)
        fallback = self.agent._fallback_plan(state)
        payload = json.dumps(
            {
                "action": "retrieve",
                "ask_attribute": None,
                "semantic_query": "blue walking shoes below eighty dollars",
                "hard_constraints": ["blue", "under $80"],
                "soft_preferences": ["walking"],
                "excluded_terms": [],
                "confidence": 0.9,
            }
        )
        with (
            patch.object(agent_module, "LLM_ENABLED", True),
            patch.object(agent_module, "_call_llm", return_value=(payload, {
                "input_tokens": 12, "output_tokens": 8
            })),
        ):
            plan, usage = self.agent._plan_with_llm(state, fallback, 4)
        self.assertIsInstance(plan, SearchPlan)
        self.assertEqual(plan.action, "retrieve")
        self.assertEqual(plan.hard_constraints, ["blue", "under $80"])
        self.assertEqual(usage["input_tokens"], 12)


class EmbeddingMetadataTests(unittest.TestCase):
    def test_local_directory_and_canonical_model_are_compatible(self) -> None:
        self.assertTrue(
            _compatible_model_names(
                "data/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"
            )
        )
        self.assertFalse(
            _compatible_model_names("BAAI/bge-small-en-v1.5", "other/model")
        )


if __name__ == "__main__":
    unittest.main()
