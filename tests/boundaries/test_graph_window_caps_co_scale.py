"""Boundary gate for task #30 A2: 5-const co-scale + structure equivalence.

Background — task #30 introduces a graph-only multi-chunk extraction
window (per-collection ``graph_extraction_window_size``). When
``window_size > 1`` the per-chunk caps inherited from the single-chunk
era would silently degrade extraction quality (a 3-chunk window
produces ~3× the entity / relation candidates but the LLM is still
capped at 32 / 32 / 60s). A2 (this PR) scales these caps linearly with
``len(window.chunk_ids)`` at the ``_extract_one_window()`` call site.

Two contracts this test enforces:

1. **window_size=1 structure equivalence (spec § 6.1, BLOCKER 1
   修法)** — when ``window_size=1`` the scaled values equal the legacy
   single-chunk values byte-for-byte. ``window_size=1`` is the
   default and the only safe ``Phase A → A3 merge`` configuration; if
   structure equivalence breaks, a tenant who never opted into
   multi-chunk windows would silently see different behaviour from
   pre-task-#30 main.
2. **5-const co-scale formula correctness (spec § 3.1.2)** — when
   ``window_size > 1`` the four CPU-bounded caps (entities / relations
   / timeout / bootstrap) and the prompt-token guardrail
   (``MAX_PROMPT_TOKENS``) all follow the spec formulas. A future PR
   that touches ``_scaled_*`` / ``_bootstrap_window_count`` /
   ``_estimate_window_prompt_tokens`` without updating both the
   formula and this test would silently regress the co-scale invariant.

Why a boundary test (not just a unit test): per Lesson #13 v3 sediment
(``docs/zh-CN/architecture/task-17-cr-review-checklist.md`` § 四), a
boundary test should cover *contracts that can drift* — the 5-const
formula is exactly that. ``_DEFAULT_MAX_ENTITIES_PER_CHUNK = 32`` etc.
are facts of the codebase already; this test does not duplicate that
fact. It pins the scaling *function*, not the constants.
"""

from __future__ import annotations

from aperag.indexing.graph_extractor import (
    _BOOTSTRAP_CHUNK_COUNT,
    _BOOTSTRAP_WINDOW_COUNT_MIN,
    _DEFAULT_MAX_ENTITIES_PER_CHUNK,
    _DEFAULT_MAX_PROMPT_TOKENS,
    _DEFAULT_MAX_RELATIONS_PER_CHUNK,
    _DEFAULT_PER_CHUNK_TIMEOUT_SECONDS,
    _bootstrap_window_count,
    _estimate_window_prompt_tokens,
    _scaled_max_entities,
    _scaled_max_relations,
    _scaled_timeout,
)

# ---------------------------------------------------------------------------
# Const #1 + #2: max_entities / max_relations co-scale
# ---------------------------------------------------------------------------


def test_window_size_1_max_entities_byte_equivalent_to_legacy():
    """window_size=1 must return the legacy per-chunk cap unchanged."""
    assert _scaled_max_entities(_DEFAULT_MAX_ENTITIES_PER_CHUNK, 1) == _DEFAULT_MAX_ENTITIES_PER_CHUNK


def test_window_size_n_max_entities_scales_linearly():
    """window_size=N must allow N× the legacy per-chunk entity cap."""
    base = _DEFAULT_MAX_ENTITIES_PER_CHUNK
    assert _scaled_max_entities(base, 2) == base * 2
    assert _scaled_max_entities(base, 3) == base * 3
    assert _scaled_max_entities(base, 5) == base * 5


def test_window_size_1_max_relations_byte_equivalent_to_legacy():
    assert _scaled_max_relations(_DEFAULT_MAX_RELATIONS_PER_CHUNK, 1) == _DEFAULT_MAX_RELATIONS_PER_CHUNK


def test_window_size_n_max_relations_scales_linearly():
    base = _DEFAULT_MAX_RELATIONS_PER_CHUNK
    assert _scaled_max_relations(base, 2) == base * 2
    assert _scaled_max_relations(base, 3) == base * 3
    assert _scaled_max_relations(base, 5) == base * 5


def test_scaled_caps_handle_zero_window_chunk_count_gracefully():
    """Defensive: an empty window should never poison the cap formula.

    This should be unreachable in production — A1's
    ``_build_graph_chunk_windows`` cannot construct an empty window —
    but if a malformed window ever reaches the scale function we want
    the legacy single-chunk cap rather than a zero or negative cap
    (which would silently drop every entity).
    """
    base = _DEFAULT_MAX_ENTITIES_PER_CHUNK
    assert _scaled_max_entities(base, 0) == base
    assert _scaled_max_relations(base, 0) == base
    assert _scaled_max_entities(base, -1) == base


# ---------------------------------------------------------------------------
# Const #3: timeout co-scale
# ---------------------------------------------------------------------------


def test_window_size_1_timeout_byte_equivalent_to_legacy():
    """window_size=1 must keep the 60s legacy timeout unchanged."""
    assert _scaled_timeout(_DEFAULT_PER_CHUNK_TIMEOUT_SECONDS, 1) == _DEFAULT_PER_CHUNK_TIMEOUT_SECONDS


def test_window_size_n_timeout_scales_linearly():
    """First-version linear formula per spec § 3.1.2 (Phase B may
    revisit to ``base × sqrt(window_size)`` once benchmark data lands).
    """
    base = _DEFAULT_PER_CHUNK_TIMEOUT_SECONDS
    assert _scaled_timeout(base, 2) == base * 2
    assert _scaled_timeout(base, 3) == base * 3
    assert _scaled_timeout(base, 5) == base * 5


def test_scaled_timeout_returns_float():
    """``asyncio.wait_for(timeout=...)`` accepts float seconds; the
    scaled timeout must remain float to avoid int-overflow surprises
    on extreme window sizes."""
    scaled = _scaled_timeout(60.0, 3)
    assert isinstance(scaled, float)
    assert scaled == 180.0


# ---------------------------------------------------------------------------
# Const #4: bootstrap window count co-scale
# ---------------------------------------------------------------------------


def test_window_size_1_bootstrap_count_equals_legacy_chunk_count():
    """window_size=1 means windows == chunks → bootstrap count must
    equal the legacy 20-chunk loop length byte-for-byte. This is the
    structural-equivalence anchor for the W11 dynamic-types feedback
    loop."""
    assert _bootstrap_window_count(1) == _BOOTSTRAP_CHUNK_COUNT


def test_bootstrap_window_count_scales_inversely_with_window_size():
    """Bootstrap should run ~20 chunks worth of serial work regardless
    of window_size. ``ceil(20 / window_size)`` keeps total bootstrap
    chunks within ±window_size of 20."""
    # window_size=2 → ceil(20/2)=10 windows = 20 chunks (matches legacy)
    assert _bootstrap_window_count(2) == 10
    # window_size=3 → ceil(20/3)=7 windows = 21 chunks (close to legacy)
    assert _bootstrap_window_count(3) == 7
    # window_size=4 → ceil(20/4)=5 windows = 20 chunks (matches legacy)
    assert _bootstrap_window_count(4) == 5
    # window_size=5 → ceil(20/5)=4 windows = 20 chunks (matches legacy)
    assert _bootstrap_window_count(5) == 4
    # window_size=10 → ceil(20/10)=2 windows = 20 chunks (matches legacy)
    assert _bootstrap_window_count(10) == 2


def test_bootstrap_window_count_floors_at_minimum():
    """A tiny document with window_size larger than the legacy chunk
    count must still run at least one bootstrap window so the W11
    feedback loop has a chance to seed the active type list."""
    # window_size=25 → ceil(20/25)=1 → returns floor 1
    assert _bootstrap_window_count(25) == _BOOTSTRAP_WINDOW_COUNT_MIN
    # window_size=100 → still 1
    assert _bootstrap_window_count(100) == _BOOTSTRAP_WINDOW_COUNT_MIN


def test_bootstrap_window_count_handles_zero_window_size_defensively():
    """A misconfigured ``window_size <= 0`` must not crash bootstrap;
    fall back to the floor value rather than divide-by-zero."""
    assert _bootstrap_window_count(0) == _BOOTSTRAP_WINDOW_COUNT_MIN
    assert _bootstrap_window_count(-1) == _BOOTSTRAP_WINDOW_COUNT_MIN


# ---------------------------------------------------------------------------
# Const #5: max_prompt_tokens guardrail co-scale (Bryce concern 3)
# ---------------------------------------------------------------------------


def test_default_max_prompt_tokens_covers_window_size_5_with_400_token_chunks():
    """The default 32k ceiling must comfortably fit the realistic
    Phase A benchmark matrix (window_size up to 5, chunk_size 400)."""
    assert _estimate_window_prompt_tokens(5) < _DEFAULT_MAX_PROMPT_TOKENS


def test_estimate_window_prompt_tokens_scales_linearly_with_window_size():
    """Per-chunk overhead + chunk content scales linearly; the prompt
    envelope (template + few-shot) is a fixed cost. This invariant
    must hold for the cap-overflow check in the bootstrap + main pass
    to behave deterministically."""
    base = _estimate_window_prompt_tokens(1)
    # Each additional window chunk adds (chunk_size + per_chunk_overhead) tokens
    delta_per_chunk = _estimate_window_prompt_tokens(2) - base
    assert delta_per_chunk > 0
    assert _estimate_window_prompt_tokens(3) == base + delta_per_chunk * 2
    assert _estimate_window_prompt_tokens(5) == base + delta_per_chunk * 4


def test_estimate_window_prompt_tokens_handles_empty_window():
    """An empty window has zero prompt cost (in practice unreachable
    but defensive against future refactors)."""
    assert _estimate_window_prompt_tokens(0) == 0
    assert _estimate_window_prompt_tokens(-1) == 0


def test_estimate_window_prompt_tokens_warns_pathological_config():
    """A pathological ``chunk_size=4000`` × ``window_size=10`` config
    must render to a number well above the default 32k ceiling so
    the cap-overflow guard fires deterministically. 4000-token chunks
    × 10-window = ~40.5k tokens, comfortably above the 32k Qwen
    context-window safety floor that the default ``MAX_PROMPT_TOKENS``
    targets."""
    pathological = _estimate_window_prompt_tokens(10, base_chunk_size=4000)
    assert pathological > _DEFAULT_MAX_PROMPT_TOKENS


# ---------------------------------------------------------------------------
# Cross-const structure equivalence (spec § 6.1 BLOCKER 1)
# ---------------------------------------------------------------------------


def test_window_size_1_full_structure_equivalence():
    """When ``window_size=1`` is set on a collection, ALL FIVE scaled
    values must equal their legacy single-chunk counterparts. This is
    the headline structure-equivalence contract: a tenant who has not
    opted into multi-chunk windows must see byte-identical scaled
    behaviour to pre-task-#30 main."""
    assert _scaled_max_entities(_DEFAULT_MAX_ENTITIES_PER_CHUNK, 1) == _DEFAULT_MAX_ENTITIES_PER_CHUNK
    assert _scaled_max_relations(_DEFAULT_MAX_RELATIONS_PER_CHUNK, 1) == _DEFAULT_MAX_RELATIONS_PER_CHUNK
    assert _scaled_timeout(_DEFAULT_PER_CHUNK_TIMEOUT_SECONDS, 1) == _DEFAULT_PER_CHUNK_TIMEOUT_SECONDS
    assert _bootstrap_window_count(1) == _BOOTSTRAP_CHUNK_COUNT
    # MAX_PROMPT_TOKENS does not scale with window_size — it is a
    # provider-context-window guardrail. We assert the default is large
    # enough that window_size=1 never trips it for the realistic
    # 400-token chunk size.
    assert _estimate_window_prompt_tokens(1) < _DEFAULT_MAX_PROMPT_TOKENS
