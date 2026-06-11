"""
tests/test_prompts.py — Unit tests for prompts.py

Covers:
  - count_tokens
  - PromptRenderer.render and missing_placeholders
  - TOKEN_BUDGET structure
  - enforce_token_budget — all trim steps including the M6 fix (Step 1b):
      Step 1   — business_description stripped
      Step 1b  — column descriptions stripped (M6: was missing before the fix)
      Step 2   — session history trimmed
      Step 3   — chunk list reduced to 3

NOTE: the Step 1b tests require the refactored prompts.py (with the M6 fix).
      They will fail on the original file intentionally — that confirms the
      fix is necessary. Replace prompts.py before running this file.

Dependencies: all in requirements.txt (tiktoken, google-genai, groq).
No network calls. No LLM instantiation. No mocks beyond TOKEN_BUDGET patching.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import prompts
from interfaces import SchemaChunk, SchemaColumn
from prompts import (
    NL_TO_SQL_SYSTEM_PROMPT,
    TOKEN_BUDGET,
    PromptRenderer,
    count_tokens,
    enforce_token_budget,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TINY = 50  # token budget small enough to force trimming on any real input


def _col(
    name: str = "amount",
    col_type: str = "DECIMAL(12,2)",
    nullable: bool = True,
    description: str | None = None,
) -> SchemaColumn:
    return SchemaColumn(name=name, type=col_type, nullable=nullable, description=description)


def _chunk(
    table: str = "claims",
    schema_id: str = "test",
    business_description: str | None = None,
    col_descriptions: list[str | None] | None = None,
    n_cols: int = 2,
) -> SchemaChunk:
    if col_descriptions is None:
        col_descriptions = [None] * n_cols
    columns = [_col(name=f"col_{i}", description=d) for i, d in enumerate(col_descriptions)]
    return SchemaChunk(
        table=table,
        schema_id=schema_id,
        business_description=business_description,
        columns=columns,
    )


def _turn(i: int = 0) -> dict:
    return {
        "nl_query": f"How many rows in table_{i}?",
        "generated_code": f"SELECT COUNT(*) FROM table_{i} LIMIT 10000",
        "insight": f"There are {i * 100} rows.",
    }


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string_is_zero(self):
        assert count_tokens("") == 0

    def test_non_empty_returns_positive(self):
        assert count_tokens("hello world") > 0

    def test_longer_text_more_tokens(self):
        assert count_tokens("word " * 100) > count_tokens("word")

    def test_return_type_is_int(self):
        assert isinstance(count_tokens("test"), int)

    def test_whitespace_only(self):
        # tiktoken may count whitespace tokens; just confirm it doesn't crash
        result = count_tokens("   \t\n  ")
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# PromptRenderer
# ---------------------------------------------------------------------------


class TestPromptRendererRender:
    def test_simple_substitution(self):
        assert PromptRenderer.render("Hello {name}!", name="world") == "Hello world!"

    def test_missing_placeholder_preserved(self):
        result = PromptRenderer.render("Hello {name} and {other}!", name="Alice")
        assert "{other}" in result
        assert "Alice" in result

    def test_dict_value_serialised_to_json(self):
        result = PromptRenderer.render("{data}", data={"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_list_value_serialised_to_json(self):
        result = PromptRenderer.render("{items}", items=[1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_int_converted_to_string(self):
        assert PromptRenderer.render("{n}", n=42) == "42"

    def test_string_injected_directly(self):
        assert PromptRenderer.render("{s}", s="raw text") == "raw text"

    def test_no_placeholders_returns_template_unchanged(self):
        t = "No curly braces here."
        assert PromptRenderer.render(t) == t

    def test_curly_braces_in_value_not_re_interpreted(self):
        # A value containing { } must not be re-processed as a new placeholder
        result = PromptRenderer.render("Result: {val}", val="{not_a_placeholder}")
        assert "{not_a_placeholder}" in result

    def test_multiple_placeholders(self):
        result = PromptRenderer.render("{a} and {b}", a="foo", b="bar")
        assert result == "foo and bar"


class TestPromptRendererMissingPlaceholders:
    def test_all_provided_returns_empty(self):
        assert PromptRenderer.missing_placeholders("{x} {y}", x=1, y=2) == []

    def test_detects_single_missing(self):
        missing = PromptRenderer.missing_placeholders("{x} {y}", x=1)
        assert "y" in missing
        assert "x" not in missing

    def test_detects_multiple_missing(self):
        missing = PromptRenderer.missing_placeholders("{a} {b} {c}")
        assert set(missing) == {"a", "b", "c"}

    def test_empty_template_returns_empty(self):
        assert PromptRenderer.missing_placeholders("No placeholders.") == []

    def test_returns_sorted_list(self):
        missing = PromptRenderer.missing_placeholders("{z} {a} {m}")
        assert missing == sorted(missing)


# ---------------------------------------------------------------------------
# TOKEN_BUDGET
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_total_key_present_and_positive(self):
        assert TOKEN_BUDGET["total"] > 0

    def test_required_keys_present(self):
        for key in ("schema_context", "session_history", "prompt_overhead", "total"):
            assert key in TOKEN_BUDGET, f"Missing key: {key}"

    def test_all_values_positive(self):
        for key, val in TOKEN_BUDGET.items():
            assert val > 0, f"TOKEN_BUDGET[{key!r}] must be positive"


# ---------------------------------------------------------------------------
# enforce_token_budget — return contract
# ---------------------------------------------------------------------------


class TestEnforceTokenBudgetContract:
    def test_returns_tuple_of_three(self):
        result = enforce_token_budget([_chunk()], [])
        assert len(result) == 3

    def test_third_element_is_int(self):
        _, _, total = enforce_token_budget([_chunk()], [])
        assert isinstance(total, int)

    def test_third_element_is_positive(self):
        _, _, total = enforce_token_budget([_chunk()], [])
        assert total > 0

    def test_empty_inputs_return_empty_outputs(self):
        chunks_out, hist_out, total = enforce_token_budget([], [])
        assert chunks_out == []
        assert hist_out == []

    def test_input_list_not_mutated(self):
        """enforce_token_budget must never modify the caller's lists in-place."""
        original_chunks = [_chunk(business_description="keep this")]
        original_history = [_turn(0)]
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            enforce_token_budget(original_chunks, original_history)
        # Originals unchanged
        assert original_chunks[0].business_description == "keep this"
        assert len(original_history) == 1


# ---------------------------------------------------------------------------
# enforce_token_budget — within budget (no trimming)
# ---------------------------------------------------------------------------


class TestEnforceTokenBudgetWithinBudget:
    def test_single_small_chunk_passes_through(self):
        c = _chunk(table="orders")
        chunks_out, _, _ = enforce_token_budget([c], [])
        assert chunks_out[0].table == "orders"

    def test_business_description_preserved_when_under_budget(self):
        c = _chunk(business_description="Brief note.")
        chunks_out, _, _ = enforce_token_budget([c], [])
        assert chunks_out[0].business_description == "Brief note."

    def test_column_description_preserved_when_under_budget(self):
        c = _chunk(col_descriptions=["Nullable — use IS NOT NULL guard."])
        chunks_out, _, _ = enforce_token_budget([c], [])
        assert chunks_out[0].columns[0].description == "Nullable — use IS NOT NULL guard."

    def test_history_preserved_when_under_budget(self):
        history = [_turn(i) for i in range(3)]
        _, hist_out, _ = enforce_token_budget([_chunk()], history)
        assert len(hist_out) == 3


# ---------------------------------------------------------------------------
# enforce_token_budget — Step 1: business_description stripped
# ---------------------------------------------------------------------------


class TestEnforceTokenBudgetStep1:
    def test_business_description_stripped_when_over_budget(self):
        c = _chunk(business_description="Quality note: claim_status is unreliable.")
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            chunks_out, _, _ = enforce_token_budget([c], [])
        assert chunks_out[0].business_description is None

    def test_multiple_chunks_all_stripped(self):
        chunks = [_chunk(table=f"t{i}", business_description=f"Note {i}") for i in range(3)]
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            chunks_out, _, _ = enforce_token_budget(chunks, [])
        assert all(c.business_description is None for c in chunks_out)

    def test_columns_intact_after_step1(self):
        c = _chunk(business_description="Strip me.", n_cols=2)
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            chunks_out, _, _ = enforce_token_budget([c], [])
        # Columns still present — only business_description stripped
        assert len(chunks_out[0].columns) == 2

    def test_original_chunk_object_not_mutated(self):
        original_desc = "Do not mutate."
        c = _chunk(business_description=original_desc)
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            enforce_token_budget([c], [])
        assert c.business_description == original_desc


# ---------------------------------------------------------------------------
# enforce_token_budget — Step 1b: column descriptions stripped (M6 fix)
# ---------------------------------------------------------------------------


class TestEnforceTokenBudgetStep1bColumnDescriptions:
    """
    These tests verify the M6 fix (Step 1b in enforce_token_budget).
    They will FAIL on the original prompts.py and PASS on the refactored one.
    """

    def test_column_descriptions_stripped_when_over_budget(self):
        # No business_description → Step 1 is a no-op.
        # Column description present → Step 1b must strip it.
        c = _chunk(
            business_description=None,
            col_descriptions=["Null rate 18 %. Use COALESCE before aggregating."],
        )
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            chunks_out, _, _ = enforce_token_budget([c], [])
        assert chunks_out[0].columns[0].description is None

    def test_column_name_type_preserved_after_description_stripped(self):
        c = _chunk(
            business_description=None,
            col_descriptions=["Some important semantic note about this column."],
        )
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            chunks_out, _, _ = enforce_token_budget([c], [])
        col = chunks_out[0].columns[0]
        # Name and type survive — only description is stripped
        assert col.name == "col_0"
        assert col.type == "DECIMAL(12,2)"

    def test_all_column_descriptions_stripped_across_chunks(self):
        chunks = [
            _chunk(table=f"t{i}", col_descriptions=[f"desc_{i}_{j}" for j in range(2)])
            for i in range(3)
        ]
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            chunks_out, _, _ = enforce_token_budget(chunks, [])
        for c in chunks_out:
            for col in c.columns:
                assert col.description is None, (
                    f"Expected col.description=None on {c.table}.{col.name}, "
                    f"got {col.description!r}"
                )

    def test_step1b_fires_after_step1_when_still_over_budget(self):
        # Both business_description AND column descriptions present.
        # Step 1 strips business_description; Step 1b strips column descriptions.
        c = _chunk(
            business_description="Business note.",
            col_descriptions=["Column note."],
        )
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            chunks_out, _, _ = enforce_token_budget([c], [])
        assert chunks_out[0].business_description is None
        assert chunks_out[0].columns[0].description is None

    def test_none_column_descriptions_are_no_op(self):
        c = _chunk(business_description=None, col_descriptions=[None, None])
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            chunks_out, _, _ = enforce_token_budget([c], [])
        # Step 1b runs but has nothing to strip — columns unchanged
        assert all(col.description is None for col in chunks_out[0].columns)
        assert len(chunks_out[0].columns) == 2


# ---------------------------------------------------------------------------
# enforce_token_budget — Step 2: history trimmed
# ---------------------------------------------------------------------------


class TestEnforceTokenBudgetStep2History:
    def test_history_trimmed_toward_five(self):
        history = [_turn(i) for i in range(10)]
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            _, hist_out, _ = enforce_token_budget([_chunk()], history)
        assert len(hist_out) <= 5

    def test_most_recent_turns_retained(self):
        history = [_turn(i) for i in range(10)]
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            _, hist_out, _ = enforce_token_budget([_chunk()], history)
        if hist_out:
            # The last original turn should always be in the retained slice
            assert history[-1] in hist_out

    def test_empty_history_unaffected(self):
        with patch.dict(prompts.TOKEN_BUDGET, {"total": _TINY}):
            _, hist_out, _ = enforce_token_budget([_chunk()], [])
        assert hist_out == []

    def test_short_history_preserved(self):
        history = [_turn(0), _turn(1)]
        # Real budget — two small turns should be fine
        _, hist_out, _ = enforce_token_budget([_chunk()], history)
        assert len(hist_out) == 2


# ---------------------------------------------------------------------------
# enforce_token_budget — Step 3: chunk list reduced to 3
# ---------------------------------------------------------------------------


class TestEnforceTokenBudgetStep3Chunks:
    def test_chunks_capped_at_three(self):
        chunks = [_chunk(table=f"t{i}") for i in range(6)]
        with patch.dict(prompts.TOKEN_BUDGET, {"total": 20}):
            chunks_out, _, _ = enforce_token_budget(chunks, [])
        assert len(chunks_out) <= 3

    def test_first_chunks_retained(self):
        chunks = [_chunk(table=f"table_{i}") for i in range(5)]
        with patch.dict(prompts.TOKEN_BUDGET, {"total": 20}):
            chunks_out, _, _ = enforce_token_budget(chunks, [])
        if len(chunks_out) == 3:
            assert chunks_out[0].table == "table_0"
            assert chunks_out[1].table == "table_1"
            assert chunks_out[2].table == "table_2"

    def test_three_or_fewer_chunks_unchanged(self):
        chunks = [_chunk(table=f"t{i}") for i in range(3)]
        # Real budget — three tiny chunks should pass through
        chunks_out, _, _ = enforce_token_budget(chunks, [])
        assert len(chunks_out) == 3


# ---------------------------------------------------------------------------
# NL_TO_SQL_SYSTEM_PROMPT sanity checks
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_is_non_empty_string(self):
        assert isinstance(NL_TO_SQL_SYSTEM_PROMPT, str)
        assert len(NL_TO_SQL_SYSTEM_PROMPT) > 200

    def test_references_schema_context(self):
        assert "schema_context" in NL_TO_SQL_SYSTEM_PROMPT

    def test_contains_limit_clause_rule(self):
        assert "LIMIT" in NL_TO_SQL_SYSTEM_PROMPT

    def test_contains_no_select_star_rule(self):
        assert (
            "SELECT *" in NL_TO_SQL_SYSTEM_PROMPT or "select *" in NL_TO_SQL_SYSTEM_PROMPT.lower()
        )

    def test_contains_unresolved_reference_error_code(self):
        assert "UNRESOLVED_REFERENCE" in NL_TO_SQL_SYSTEM_PROMPT
