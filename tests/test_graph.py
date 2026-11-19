"""Red tests for the render-graph bugfix exercise."""

import pytest

from rendergraph.graph import (
    GraphError,
    alias_resources,
    cull_unused,
    estimate_memory,
    execute_batch,
    insert_barriers,
    merge_passes,
    toposort,
    validate,
)


def test_toposort_respects_writes_before_reads():
    passes = [
        {"name": "present", "reads": ["color"], "writes": []},
        {"name": "gbuffer", "reads": [], "writes": ["albedo"]},
        {"name": "lighting", "reads": ["albedo"], "writes": ["color"]},
    ]

    assert toposort(passes) == ["gbuffer", "lighting", "present"]


def test_validate_detects_cycle():
    graph = {
        "passes": [
            {"name": "a", "reads": ["b_out"], "writes": ["a_out"]},
            {"name": "b", "reads": ["a_out"], "writes": ["b_out"]},
        ],
        "resources": {
            "a_out": {"size": [64, 64], "format": "rgba8"},
            "b_out": {"size": [64, 64], "format": "rgba8"},
        },
    }

    issues = validate(graph)

    assert any("cycle" in issue.lower() for issue in issues)


def test_alias_keeps_overlapping_lifetimes_separate():
    graph = {
        "passes": [],
        "resources": {
            "rt_a": {"size": [1920, 1080], "format": "rgba16f"},
            "rt_b": {"size": [1920, 1080], "format": "rgba16f"},
        },
    }
    lifetimes = {"rt_a": (0, 4), "rt_b": (2, 6)}

    result = alias_resources(graph, lifetimes)

    assert result["alias"]["rt_a"] != result["alias"]["rt_b"]
    assert result["count"] == 2


def test_barriers_inserted_for_same_format_read_write():
    passes = [
        {"name": "draw", "reads": [], "writes": ["hdr"]},
        {"name": "resolve", "reads": ["hdr"], "writes": []},
    ]
    formats = {("draw", "hdr"): "rgba16f", ("resolve", "hdr"): "rgba16f"}

    assert insert_barriers(["draw", "resolve"], passes, formats) == [
        ("draw", "resolve", "hdr")
    ]

    readers = [
        {"name": "encode", "reads": ["hdr"], "writes": []},
        {"name": "decode", "reads": ["hdr"], "writes": []},
    ]
    changed = {("encode", "hdr"): "rgba16f", ("decode", "hdr"): "rgba8"}

    assert insert_barriers(["encode", "decode"], readers, changed) == [
        ("encode", "decode", "hdr")
    ]


def test_estimate_memory_counts_aliases_once():
    graph = {
        "passes": [],
        "resources": {
            "rt_a": {"size": [800, 600], "format": "rgba8"},
            "rt_b": {"size": [800, 600], "format": "rgba8"},
            "depth": {"size": [800, 600], "format": "d32"},
        },
        "alias": {
            "rt_a": "phys_color",
            "rt_b": "phys_color",
            "depth": "phys_depth",
        },
    }
    sizes = {"rt_a": 1024, "rt_b": 1024, "depth": 512}

    assert estimate_memory(graph, sizes) == 1536


def test_cull_keeps_transitive_dependencies():
    graph = {
        "passes": [
            {"name": "gbuffer", "reads": [], "writes": ["albedo"]},
            {"name": "debug", "reads": ["albedo"], "writes": ["overlay"]},
            {"name": "lighting", "reads": ["albedo"], "writes": ["color"]},
            {"name": "present", "reads": ["color"], "writes": ["frame"]},
        ],
        "resources": {
            "albedo": {"size": [1280, 720], "format": "rgba16f"},
            "overlay": {"size": [1280, 720], "format": "rgba8"},
            "color": {"size": [1280, 720], "format": "rgba16f"},
            "frame": {"size": [1280, 720], "format": "rgba8"},
        },
    }

    result = cull_unused(graph, ["frame"])

    assert result == {
        "kept": ["gbuffer", "lighting", "present"],
        "pruned": ["debug"],
    }


def test_merge_rejects_write_read_conflict():
    graph = {
        "passes": [
            {"name": "gbuffer", "reads": [], "writes": ["albedo"]},
            {"name": "lighting", "reads": ["albedo"], "writes": ["color"]},
        ],
        "resources": {
            "albedo": {"size": [1280, 720], "format": "rgba16f"},
            "color": {"size": [1280, 720], "format": "rgba16f"},
        },
    }

    assert merge_passes(graph, "gbuffer", "lighting") is False

    mergeable = {
        "passes": [
            {"name": "clear", "reads": [], "writes": ["depth"]},
            {"name": "ui", "reads": [], "writes": ["overlay"]},
        ],
        "resources": {
            "depth": {"size": [1280, 720], "format": "d32"},
            "overlay": {"size": [1280, 720], "format": "rgba8"},
        },
    }

    assert merge_passes(mergeable, "clear", "ui") is True


def test_validate_reports_undeclared_resource():
    graph = {
        "passes": [
            {"name": "lighting", "reads": ["shadow_map"], "writes": ["color"]},
        ],
        "resources": {
            "color": {"size": [1280, 720], "format": "rgba16f"},
        },
    }

    issues = validate(graph)

    assert any(
        "undeclared resource" in issue and "shadow_map" in issue for issue in issues
    )


def test_toposort_same_layer_is_name_stable():
    passes = [
        {"name": "zeta", "reads": [], "writes": ["z_tex"]},
        {"name": "alpha", "reads": [], "writes": ["a_tex"]},
    ]

    first = toposort(passes)
    second = toposort(list(reversed(passes)))

    assert first == ["alpha", "zeta"]
    assert second == ["alpha", "zeta"]


def test_execute_batch_rolls_back_on_invalid_graph():
    graphs = [
        {
            "name": "good",
            "passes": [{"name": "paint", "reads": [], "writes": ["frame"]}],
            "resources": {
                "frame": {"size": [1280, 720], "format": "rgba8"},
            },
        },
        {
            "name": "bad",
            "passes": [{"name": "sample", "reads": ["missing"], "writes": ["frame"]}],
            "resources": {
                "frame": {"size": [1280, 720], "format": "rgba8"},
            },
        },
    ]

    result = execute_batch(graphs, {"good": ["frame"], "bad": ["frame"]})

    assert result["ok"] is False
    assert result["results"] == {}
    assert any(
        "undeclared resource" in issue for issue in result["errors"]["bad"]
    )


def test_single_pass_toposort_returns_itself():
    passes = [{"name": "only", "reads": [], "writes": ["target"]}]

    assert toposort(passes) == ["only"]


def test_empty_toposort_returns_empty():
    assert toposort([]) == []


def test_cycle_raises_graph_error():
    passes = [
        {"name": "a", "reads": ["b_out"], "writes": ["a_out"]},
        {"name": "b", "reads": ["a_out"], "writes": ["b_out"]},
    ]

    with pytest.raises(GraphError):
        toposort(passes)


def test_culled_order_matches_toposort():
    graph = {
        "passes": [
            {"name": "mid", "reads": ["base_tex"], "writes": ["final"]},
            {"name": "zeta", "reads": [], "writes": ["base_tex"]},
            {"name": "alpha", "reads": [], "writes": ["dead"]},
        ],
        "resources": {
            "base_tex": {"size": [640, 360], "format": "rgba16f"},
            "final": {"size": [640, 360], "format": "rgba8"},
            "dead": {"size": [640, 360], "format": "rgba8"},
        },
    }

    result = cull_unused(graph, ["final"])
    order = toposort(graph["passes"])

    assert result["kept"] == ["zeta", "mid"]
    assert result["kept"] == [name for name in order if name in result["kept"]]
    assert result["pruned"] == ["alpha"]


def test_alias_count_at_most_lifetime_peak():
    graph = {
        "passes": [],
        "resources": {
            name: {"size": [256, 256], "format": "rgba8"}
            for name in ("a", "b", "c", "d")
        },
    }
    lifetimes = {
        "a": (0, 2),
        "b": (2, 4),
        "c": (1, 3),
        "d": (3, 5),
    }

    result = alias_resources(graph, lifetimes)

    assert result["count"] <= 2
