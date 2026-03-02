"""
Comprehensive tests for similarity cache functionality (SPEC-2.2 — T3).

Tests cover:
- Cache hit avoidance of LLM calls
- Order-agnostic cache key behavior
- Cache storage and retrieval
- Cache behavior with and without descriptions
"""

import pytest
from unittest.mock import MagicMock, patch

from ontogen import Ontology


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def ontology_with_cache(mock_agent):
    """Fresh Ontology instance with initialized similarity cache."""
    return Ontology(domain="Star Trek", agent=mock_agent)


# ============================================================================
# T3: Similarity Cache Tests (3 tests)
# ============================================================================


class TestSimilarityCache:
    """Tests for Ontology._get_similarity_cached() method."""

    def test_cache_hit_avoids_llm_call(self, ontology_with_cache, mock_agent):
        """Verify that a cached pair uses stored result without calling LLM.

        Scenario:
        - Pre-populate cache with ("Vulcans", "Spock") → 85
        - Call _get_similarity_cached("Vulcans", "Spock")
        - Verify: Returns 85, LLM not called again

        This tests the fundamental cache hit path.
        """
        # Setup: Pre-populate cache
        ontology_with_cache.similarity_cache[("Spock", "Vulcans")] = 85.0

        # Setup: Configure mock agent to fail if called (to prove cache avoids it)
        mock_agent.get_similarity.side_effect = RuntimeError("Should not call LLM on cache hit!")
        mock_agent.get_similarity_with_descriptions.side_effect = RuntimeError(
            "Should not call LLM on cache hit!"
        )
        ontology_with_cache.agent = mock_agent

        # Call: Request same pair (should hit cache)
        result = ontology_with_cache._get_similarity_cached("Vulcans", "Spock")

        # Assert: Returns cached value, no LLM call
        assert result == 85.0
        mock_agent.get_similarity.assert_not_called()
        mock_agent.get_similarity_with_descriptions.assert_not_called()

    def test_cache_key_is_sorted(self, ontology_with_cache, mock_agent):
        """Verify that cache key is order-agnostic.

        Scenario:
        - Pre-populate cache with ("Humans", "Vulcans") → 65
        - Call _get_similarity_cached("Vulcans", "Humans")
        - Verify: Returns 65 (same entry, despite reversed argument order)
        - Verify: LLM is not called

        This tests that (A, B) and (B, A) map to the same cache key via tuple(sorted([...]))
        """
        # Setup: Pre-populate cache with one order
        ontology_with_cache.similarity_cache[("Humans", "Vulcans")] = 65.0

        # Setup: Configure mock to fail if called
        mock_agent.get_similarity.side_effect = RuntimeError("Should not call LLM!")
        mock_agent.get_similarity_with_descriptions.side_effect = RuntimeError("Should not call LLM!")
        ontology_with_cache.agent = mock_agent

        # Call: Request the pair in reverse order
        result = ontology_with_cache._get_similarity_cached("Vulcans", "Humans")

        # Assert: Should hit the same cache entry
        assert result == 65.0
        mock_agent.get_similarity.assert_not_called()
        mock_agent.get_similarity_with_descriptions.assert_not_called()

    def test_cache_stores_result_after_llm_call(self, ontology_with_cache, mock_agent):
        """Verify that LLM results are stored in cache for future lookups.

        Scenario:
        - Cache is empty
        - Call _get_similarity_cached("Species", "Vulcans") with no descriptions
        - LLM returns {"similarity": 72}
        - Verify: Result is stored in cache under sorted key
        - Call _get_similarity_cached("Vulcans", "Species") (reversed order)
        - Verify: Returns 72 from cache, no new LLM call

        This tests the cache miss → store → hit flow.
        """
        # Setup: Cache is empty (no pre-population)
        assert len(ontology_with_cache.similarity_cache) == 0

        # Setup: Configure mock to return a value on first call
        mock_agent.get_similarity_with_descriptions.return_value = {"similarity": 72}
        ontology_with_cache.agent = mock_agent

        # Call 1: First call should hit LLM (cache miss)
        result_1 = ontology_with_cache._get_similarity_cached("Species", "Vulcans")

        # Assert: Returns the LLM result
        assert result_1 == 72
        # Assert: LLM was called once
        mock_agent.get_similarity_with_descriptions.assert_called_once()
        # Assert: Result is now in cache
        cache_key = tuple(sorted(["Species", "Vulcans"]))
        assert cache_key in ontology_with_cache.similarity_cache
        assert ontology_with_cache.similarity_cache[cache_key] == 72

        # Reset mock call count
        mock_agent.reset_mock()
        mock_agent.get_similarity_with_descriptions.side_effect = RuntimeError("Should not call LLM on second lookup!")

        # Call 2: Second call with reversed order should hit cache
        result_2 = ontology_with_cache._get_similarity_cached("Vulcans", "Species")

        # Assert: Returns same cached value
        assert result_2 == 72
        # Assert: LLM was not called (cache hit)
        mock_agent.get_similarity_with_descriptions.assert_not_called()

    def test_cache_with_descriptions_uses_correct_method(self, ontology_with_cache, mock_agent):
        """Verify that _get_similarity_cached uses get_similarity_with_descriptions when descriptions provided.

        Scenario:
        - Cache is empty
        - Call _get_similarity_cached with descriptions for both terms
        - Verify: Uses get_similarity_with_descriptions() method, not get_similarity()
        - Verify: Result is cached under sorted key

        This tests the LLM method selection path (with descriptions).
        """
        # Setup: Cache is empty
        assert len(ontology_with_cache.similarity_cache) == 0

        # Setup: Configure mocks
        mock_agent.get_similarity.side_effect = RuntimeError("Should use with_descriptions method!")
        mock_agent.get_similarity_with_descriptions.return_value = {
            "term_x": "Spock",
            "description_x": "Half-Vulcan Starfleet officer",
            "term_y": "Vulcans",
            "description_y": "Logical, telepathic species",
            "similarity": 78,
        }
        ontology_with_cache.agent = mock_agent

        # Call: With descriptions
        result = ontology_with_cache._get_similarity_cached(
            term_a="Spock",
            term_b="Vulcans",
            description_a="Half-Vulcan Starfleet officer",
            description_b="Logical, telepathic species",
        )

        # Assert: Returned the correct score
        assert result == 78
        # Assert: Used the correct method
        mock_agent.get_similarity_with_descriptions.assert_called_once()
        mock_agent.get_similarity.assert_not_called()
        # Assert: Cached the result
        cache_key = tuple(sorted(["Spock", "Vulcans"]))
        assert ontology_with_cache.similarity_cache[cache_key] == 78

    def test_cache_handles_none_similarity(self, ontology_with_cache, mock_agent):
        """Verify defensive parsing when LLM returns None for similarity.

        Scenario:
        - Cache is empty
        - LLM returns {"similarity": None}
        - Verify: Defaults to 0.0 and caches it

        This tests error handling for malformed LLM responses.
        """
        # Setup: Mock returns None similarity
        mock_agent.get_similarity_with_descriptions.return_value = {"similarity": None}
        ontology_with_cache.agent = mock_agent

        # Call
        result = ontology_with_cache._get_similarity_cached("TermA", "TermB")

        # Assert: Returns 0.0 (default)
        assert result == 0.0
        # Assert: Cached the default
        cache_key = tuple(sorted(["TermA", "TermB"]))
        assert ontology_with_cache.similarity_cache[cache_key] == 0.0

    def test_cache_with_partial_descriptions(self, ontology_with_cache, mock_agent):
        """Verify behavior when only one term has a description.

        Scenario:
        - Call _get_similarity_cached("Spock", "Vulcans", description_a="...", description_b=None)
        - Verify: Uses get_similarity_with_descriptions() (because at least one description exists)
        - Verify: Result is cached

        This tests the "partial descriptions" path (at least one present → use with_descriptions method).
        """
        # Setup: Mock
        mock_agent.get_similarity.side_effect = RuntimeError("Should use with_descriptions!")
        mock_agent.get_similarity_with_descriptions.return_value = {"similarity": 55}
        ontology_with_cache.agent = mock_agent

        # Call: description_a present, description_b absent
        result = ontology_with_cache._get_similarity_cached(
            term_a="Spock",
            term_b="Vulcans",
            description_a="Half-Vulcan officer",
            description_b=None,
        )

        # Assert: Used with_descriptions method
        assert result == 55
        mock_agent.get_similarity_with_descriptions.assert_called_once()
        mock_agent.get_similarity.assert_not_called()

    def test_multiple_cache_entries(self, ontology_with_cache, mock_agent):
        """Verify that cache correctly maintains multiple independent entries.

        Scenario:
        - Pre-populate cache with two pairs: (A, B) → 60, (C, D) → 80
        - Call _get_similarity_cached("A", "B")
        - Call _get_similarity_cached("C", "D")
        - Verify: Both return correct cached values
        - Verify: Cache has exactly 2 entries

        This tests cache isolation between different pairs.
        """
        # Setup: Pre-populate with two entries
        ontology_with_cache.similarity_cache[("A", "B")] = 60.0
        ontology_with_cache.similarity_cache[("C", "D")] = 80.0

        # Setup: Mock to fail if called
        mock_agent.get_similarity.side_effect = RuntimeError("Should not call LLM!")
        ontology_with_cache.agent = mock_agent

        # Call: Both pairs
        result_1 = ontology_with_cache._get_similarity_cached("A", "B")
        result_2 = ontology_with_cache._get_similarity_cached("C", "D")

        # Assert: Correct values
        assert result_1 == 60.0
        assert result_2 == 80.0
        # Assert: Cache size unchanged
        assert len(ontology_with_cache.similarity_cache) == 2
        # Assert: No LLM calls
        mock_agent.get_similarity.assert_not_called()
