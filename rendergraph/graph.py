"""Graph contracts for the render-graph bugfix exercise.

Graph data model
----------------
A graph is an inline dictionary::

    {
        "name": "frame",                       # optional outside execute_batch
        "passes": [
            {"name": "gbuffer", "reads": [], "writes": ["albedo"]},
            {"name": "lighting", "reads": ["albedo"], "writes": ["color"]},
        ],
        "resources": {
            "albedo": {"size": [1920, 1080], "format": "rgba16f"},
            "color": {"size": [1920, 1080], "format": "rgba16f"},
        },
        "limits": {"max_width": 4096, "max_height": 4096},  # optional
        "alias": {"albedo": "albedo", "color": "color"},     # optional
    }

Pass names and resource ids are non-empty strings. Pass names are unique
inside one graph. ``reads`` and ``writes`` are ordered sequences of resource
ids; an empty sequence is allowed.

Unless a function says otherwise, all ordering is deterministic: hard
dependencies are always respected, and among passes that are simultaneously
ready the lexicographically smallest name is emitted first. A graph with a
dependency cycle is invalid.
"""

from __future__ import annotations


class GraphError(Exception):
    """Raised when a render graph cannot be ordered or executed."""


def toposort(passes: list[dict]) -> list[str]:
    """Return a stable dependency order for ``passes``.

    ``passes`` is the graph's list of pass dictionaries. An edge ``A -> B``
    exists when:

    * ``A`` writes a resource that ``B`` reads (read-after-write), or
    * ``A`` and ``B`` both write the same resource, in which case their
      relative registration order in ``passes`` is preserved.

    The result contains every pass name exactly once. Whenever several passes
    are ready at the same time, the lexicographically smallest name is emitted
    first, so two runs over the same input always produce the same order.

    Raises:
        GraphError: if a pass name is duplicated, a pass lacks a name, or the
            dependency graph contains a cycle.
    """

    raise NotImplementedError("toposort is not implemented")


def alias_resources(graph: dict, lifetimes: dict) -> dict:
    """Assign non-overlapping render targets to shared physical resources.

    ``lifetimes`` maps a resource id to a half-open integer interval
    ``(start, end)``; ``start`` is inclusive and ``end`` is exclusive. Only
    resources with the same ``size`` and ``format`` may share storage, and
    their intervals must not overlap (``end <= other_start``). Resources
    missing from ``lifetimes`` are treated as live forever and therefore
    cannot share storage.

    Return ``{"alias": {resource_id: physical_id}, "count": int}``. Every
    declared resource id appears in ``alias``; non-shared resources map to
    themselves. ``count`` is the number of distinct physical resources used.
    The assignment is deterministic: resources are considered by
    ``(start, end, resource_id)`` and each is placed in the first compatible
    physical slot. Aliasing must never make two live ranges overlap.
    """

    raise NotImplementedError("alias_resources is not implemented")


def insert_barriers(order: list[str], passes: list[dict], formats: dict) -> list[tuple]:
    """Return the barriers needed between adjacent passes in ``order``.

    ``order`` is a topological order of pass names and ``passes`` is the
    corresponding pass list. ``formats`` maps ``(pass_name, resource_id)`` to
    a format label; a missing entry means the format is unknown.

    For each adjacent pair ``(before, after)``, a barrier is emitted for a
    resource when either condition holds:

    * both passes touch the resource and at least one of them writes it, or
    * both passes touch the resource, both have known format labels, and the
      labels differ (a format change always needs a barrier).

    Each barrier is the tuple ``(before, after, resource_id)``. Results are
    ordered by adjacent pair and then by resource id, with duplicates removed.
    Two passes that only read the same resource need no barrier unless their
    format labels differ.
    """

    raise NotImplementedError("insert_barriers is not implemented")


def validate(graph: dict) -> list[str]:
    """Return deterministic, human-readable problems with ``graph``.

    The returned list is empty for a valid graph. It may report:

    * an undeclared resource referenced by a pass read or write,
    * a pass that lists the same resource in both ``reads`` and ``writes``,
    * a dependency cycle,
    * a resource whose ``size`` is malformed, non-positive, or exceeds
      ``graph["limits"]`` (``max_width`` / ``max_height``).

    An undeclared-resource message contains ``undeclared resource`` and the
    resource id. A cycle message contains ``cycle``. A read/write conflict
    message contains ``read/write conflict``. A bounds message contains
    ``exceeds limit`` and both dimensions. Validation never executes passes
    and never raises for ordinary bad graph data.
    """

    raise NotImplementedError("validate is not implemented")


def estimate_memory(graph: dict, sizes: dict) -> int:
    """Return the peak-ish memory total for the physical resources.

    ``sizes`` maps resource ids to byte counts; a missing entry contributes
    zero. ``graph["alias"]`` maps logical resource ids to physical ids. Every
    physical group is charged once, using the largest size among the logical
    resources mapped to it, so aliased storage is never counted twice.

    The result is an ``int`` number of bytes. If ``alias`` is absent, each
    resource is its own physical resource.
    """

    raise NotImplementedError("estimate_memory is not implemented")


def cull_unused(graph: dict, outputs: list[str]) -> dict:
    """Keep only passes needed to produce ``outputs``.

    Mark every pass that writes an output resource, then repeatedly mark every
    pass that writes a resource read by an already-marked pass. Passes that
    are never marked are pruned.

    Return ``{"kept": [...], "pruned": [...]}``. Both lists contain pass names
    in the same stable topological order that :func:`toposort` defines. The
    graph must be acyclic; otherwise ``GraphError`` is raised.
    """

    raise NotImplementedError("cull_unused is not implemented")


def merge_passes(graph: dict, a: str, b: str) -> bool:
    """Return whether passes ``a`` and ``b`` can be merged safely.

    Merging is allowed only when:

    * both names exist and are adjacent in the graph's stable topological
      order;
    * no resource is written by both passes, written by ``a`` and read by
      ``b``, or written by ``b`` and read by ``a`` (those cases would remove a
      write or need a barrier);
    * the two passes do not declare different explicit ``format`` values.

    Shared read-only resources are harmless. Unknown names, non-adjacent
    passes, cycles, and conflicts return ``False`` rather than raising.
    """

    raise NotImplementedError("merge_passes is not implemented")


def execute_batch(graphs: list[dict], outputs: dict) -> dict:
    """Execute a batch atomically and return a structured summary.

    ``graphs`` is a sequence of graph dictionaries, each with a unique
    ``"name"``. ``outputs`` maps a graph name to its list of requested output
    resource ids.

    The whole batch is validated before any graph is executed. If at least one
    graph is invalid, return::

        {"ok": False, "results": {}, "errors": {name: [issue, ...]}}

    No partial results are retained. If every graph is valid, return::

        {"ok": True,
         "results": {name: {"order": [...], "kept": [...], "pruned": [...]}},
         "errors": {}}

    ``order`` is the stable topological order and ``kept`` / ``pruned`` come
    from :func:`cull_unused`. A batch with no graphs is valid and yields empty
    ``results`` and ``errors``.
    """

    raise NotImplementedError("execute_batch is not implemented")
