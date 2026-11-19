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

import heapq

_INF = float("inf")


class GraphError(Exception):
    """Raised when a render graph cannot be ordered or executed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pass_name(pas: dict) -> str:
    """Return a validated pass name, raising ``GraphError`` on bad data."""
    if not isinstance(pas, dict):
        raise GraphError(f"pass entry is not a mapping: {pas!r}")
    name = pas.get("name")
    if not isinstance(name, str) or not name:
        raise GraphError(f"pass has an invalid name: {name!r}")
    return name


def _resource_list(pas: dict, key: str, name: str) -> list[str]:
    """Return a validated list of resource ids for ``pas[key]``."""
    raw = pas.get(key, [])
    if raw is None:
        return []
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise GraphError(f"pass {name!r} has a malformed {key!r} list")
    result = []
    for resource in raw:
        if not isinstance(resource, str) or not resource:
            raise GraphError(f"pass {name!r} has an invalid {key!r} entry: {resource!r}")
        result.append(resource)
    return result


def _prepare_passes(passes: list[dict]) -> list[dict]:
    """Normalize the pass list and reject duplicate / missing names."""
    if passes is None:
        raise GraphError("passes must be a sequence of pass mappings")
    try:
        items = list(passes)
    except TypeError as exc:  # pragma: no cover - defensive
        raise GraphError("passes must be a sequence of pass mappings") from exc

    prepared = []
    seen = set()
    for index, pas in enumerate(items):
        name = _pass_name(pas)
        if name in seen:
            raise GraphError(f"duplicate pass name: {name!r}")
        seen.add(name)
        prepared.append(
            {
                "name": name,
                "index": index,
                "reads": _resource_list(pas, "reads", name),
                "writes": _resource_list(pas, "writes", name),
            }
        )
    return prepared


def _build_edges(prepared: list[dict]) -> dict[str, set[str]]:
    """Build the dependency edges described by the :func:`toposort` contract.

    ``A -> B`` when ``A`` writes a resource ``B`` reads (read-after-write),
    and when ``A``/``B`` both write the same resource: the later writer
    depends on the earlier one so registration order is preserved.
    """
    writers: dict[str, list[dict]] = {}
    readers: dict[str, list[dict]] = {}
    for pas in prepared:
        # A pass writing the same resource twice must not gain a self-edge.
        for resource in dict.fromkeys(pas["writes"]):
            writers.setdefault(resource, []).append(pas)
        for resource in dict.fromkeys(pas["reads"]):
            readers.setdefault(resource, []).append(pas)

    successors: dict[str, set[str]] = {pas["name"]: set() for pas in prepared}

    # Write-after-write: earlier registration writes first.
    for resource, group in writers.items():
        for first, second in zip(group, group[1:]):
            successors[first["name"]].add(second["name"])

    # Read-after-write: a writer must precede every reader of the resource.
    for resource, group in writers.items():
        for writer in group:
            for reader in readers.get(resource, ()):
                if reader["name"] != writer["name"]:
                    successors[writer["name"]].add(reader["name"])

    return successors


def _stable_topo_order(prepared: list[dict]) -> list[str]:
    """Kahn's algorithm with a name heap for deterministic same-layer order."""
    successors = _build_edges(prepared)
    indegree = {pas["name"]: 0 for pas in prepared}
    for targets in successors.values():
        for target in targets:
            indegree[target] += 1

    ready = [name for name, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)

    order: list[str] = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for target in sorted(successors[name]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)

    if len(order) != len(prepared):
        remaining = sorted(set(indegree) - set(order))
        raise GraphError(f"dependency cycle detected among passes: {remaining}")
    return order


def _pas_lookup(prepared: list[dict]) -> dict[str, dict]:
    return {pas["name"]: pas for pas in prepared}


def _resource_table(graph: dict):
    resources = graph.get("resources") if isinstance(graph, dict) else None
    return resources if isinstance(resources, dict) else {}


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _graph_formats(graph: dict) -> dict[tuple[str, str], object]:
    """Collect explicit per-(pass, resource) format labels from a graph."""
    result: dict[tuple[str, str], object] = {}
    raw = graph.get("formats") if isinstance(graph, dict) else None
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(key, tuple) and len(key) == 2:
                result[key] = value
            elif isinstance(value, dict):
                for resource, label in value.items():
                    result[(key, resource)] = label
    graph_passes = graph.get("passes") if isinstance(graph, dict) else None
    for pas in graph_passes or []:
        if isinstance(pas, dict) and isinstance(pas.get("formats"), dict):
            name = pas.get("name")
            if isinstance(name, str):
                for resource, label in pas["formats"].items():
                    result[(name, resource)] = label
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    prepared = _prepare_passes(passes)
    return _stable_topo_order(prepared)


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

    resources = _resource_table(graph)
    lifetimes = lifetimes or {}

    entries = []
    for resource_id, descriptor in resources.items():
        if not isinstance(resource_id, str):
            continue
        descriptor = descriptor if isinstance(descriptor, dict) else {}
        if resource_id in lifetimes and lifetimes[resource_id] is not None:
            interval = lifetimes[resource_id]
            try:
                start, end = interval
                start, end = int(start), int(end)
            except (TypeError, ValueError) as exc:
                raise GraphError(
                    f"invalid lifetime for resource {resource_id!r}: {interval!r}"
                ) from exc
        else:
            start, end = _INF, _INF
        entries.append(
            {
                "id": resource_id,
                "size": descriptor.get("size"),
                "format": descriptor.get("format"),
                "start": start,
                "end": end,
            }
        )

    entries.sort(key=lambda item: (item["start"], item["end"], item["id"]))

    slots: list[dict] = []
    alias: dict[str, str] = {}
    for entry in entries:
        placed = False
        for slot in slots:
            if slot["size"] != entry["size"] or slot["format"] != entry["format"]:
                continue
            # A live-forever range (missing lifetime) overlaps everything and
            # must never share storage.
            overlaps = (
                entry["start"] == _INF
                or slot["start"] == _INF
                or not (
                    entry["end"] <= slot["start"]
                    or slot["end"] <= entry["start"]
                )
            )
            if not overlaps:
                slot["start"] = min(slot["start"], entry["start"])
                slot["end"] = max(slot["end"], entry["end"])
                slot["members"].append(entry["id"])
                alias[entry["id"]] = slot["physical"]
                placed = True
                break
        if not placed:
            slots.append(
                {
                    "physical": entry["id"],
                    "size": entry["size"],
                    "format": entry["format"],
                    "start": entry["start"],
                    "end": entry["end"],
                    "members": [entry["id"]],
                }
            )
            alias[entry["id"]] = entry["id"]

    return {"alias": alias, "count": len(slots)}


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

    prepared = _prepare_passes(passes)
    lookup = _pas_lookup(prepared)
    formats = formats or {}

    barriers: list[tuple] = []
    seen = set()
    for before, after in zip(order, order[1:]):
        first = lookup.get(before)
        second = lookup.get(after)
        if first is None or second is None:
            continue
        first_reads, first_writes = set(first["reads"]), set(first["writes"])
        second_reads, second_writes = set(second["reads"]), set(second["writes"])
        touched = (first_reads | first_writes) & (second_reads | second_writes)

        for resource in sorted(touched):
            hazard = (
                resource in first_writes
                or resource in second_writes
            )
            before_format = formats.get((before, resource))
            after_format = formats.get((after, resource))
            format_changed = (
                before_format is not None
                and after_format is not None
                and before_format != after_format
            )
            if not hazard and not format_changed:
                continue
            barrier = (before, after, resource)
            if barrier not in seen:
                seen.add(barrier)
                barriers.append(barrier)
    return barriers


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

    issues: list[str] = []
    if not isinstance(graph, dict):
        return ["graph is not a mapping"]

    declared = _resource_table(graph)
    declared_ids = set(declared)

    passes = graph.get("passes", [])
    if passes is None:
        passes = []
    if isinstance(passes, str) or not isinstance(passes, (list, tuple)):
        return ["passes must be a list"]
    passes = list(passes)

    # Per-pass structural checks: names, undeclared resources, conflicts.
    names: list[str] = []
    seen_names = set()
    for position, pas in enumerate(passes):
        if not isinstance(pas, dict):
            issues.append(f"pass at position {position} is not a mapping")
            continue
        name = pas.get("name")
        if not isinstance(name, str) or not name:
            issues.append(f"pass at position {position} has no valid name")
            continue
        if name in seen_names:
            issues.append(f"duplicate pass name {name!r}")
            continue
        seen_names.add(name)
        names.append(name)

        try:
            reads = _resource_list(pas, "reads", name)
            writes = _resource_list(pas, "writes", name)
        except GraphError as exc:
            issues.append(str(exc))
            continue

        for resource in reads:
            if resource not in declared_ids:
                issues.append(
                    f"pass {name!r} reads undeclared resource {resource!r}"
                )
        for resource in writes:
            if resource not in declared_ids:
                issues.append(
                    f"pass {name!r} writes undeclared resource {resource!r}"
                )
        for resource in sorted(set(reads) & set(writes)):
            issues.append(
                f"pass {name!r} has read/write conflict on resource {resource!r}"
            )

    # Cycle detection. Duplicate/invalid names make the graph unordered.
    if len(names) == len(passes):
        try:
            _stable_topo_order(_prepare_passes(passes))
        except GraphError as exc:
            issues.append(str(exc))

    # Resource shape and limits.
    limits = graph.get("limits")
    if not isinstance(limits, dict):
        limits = {}
    max_width = limits.get("max_width")
    max_height = limits.get("max_height")

    for resource_id, descriptor in declared.items():
        if not isinstance(resource_id, str):
            issues.append(f"resource id is not a string: {resource_id!r}")
            continue
        if not isinstance(descriptor, dict):
            issues.append(f"resource {resource_id!r} descriptor is malformed")
            continue
        size = descriptor.get("size")
        if (
            isinstance(size, (list, tuple))
            and len(size) == 2
            and _positive_int(size[0])
            and _positive_int(size[1])
        ):
            width, height = size
            out_of_bounds = (
                (_positive_int(max_width) and width > max_width)
                or (_positive_int(max_height) and height > max_height)
            )
            if out_of_bounds:
                issues.append(
                    f"resource {resource_id!r} size {width}x{height} "
                    f"exceeds limit {max_width}x{max_height}"
                )
            continue

        shown = (
            f"{size!r}"
            if isinstance(size, (list, tuple))
            else repr(size)
        )
        issues.append(
            f"resource {resource_id!r} has malformed size {shown}"
        )

    # De-duplicate while preserving first-seen (graph traversal) order.
    unique: list[str] = []
    seen_issues = set()
    for issue in issues:
        if issue not in seen_issues:
            seen_issues.add(issue)
            unique.append(issue)
    return unique


def estimate_memory(graph: dict, sizes: dict) -> int:
    """Return the peak-ish memory total for the physical resources.

    ``sizes`` maps resource ids to byte counts; a missing entry contributes
    zero. ``graph["alias"]`` maps logical resource ids to physical ids. Every
    physical group is charged once, using the largest size among the logical
    resources mapped to it, so aliased storage is never counted twice.

    The result is an ``int`` number of bytes. If ``alias`` is absent, each
    resource is its own physical resource.
    """

    resources = _resource_table(graph)
    sizes = sizes or {}
    raw_alias = graph.get("alias") if isinstance(graph, dict) else None
    alias = raw_alias if isinstance(raw_alias, dict) else {}

    logical_ids = list(resources)
    for resource_id in sizes:
        if resource_id not in logical_ids:
            logical_ids.append(resource_id)

    groups: dict[object, int] = {}
    for resource_id in logical_ids:
        physical = alias.get(resource_id, resource_id)
        try:
            key = physical if isinstance(physical, (str, int)) else repr(physical)
        except TypeError:  # pragma: no cover - defensive
            key = repr(physical)
        try:
            size = int(sizes.get(resource_id, 0) or 0)
        except (TypeError, ValueError):
            size = 0
        groups[key] = max(groups.get(key, 0), size)

    return int(sum(groups.values()))


def cull_unused(graph: dict, outputs: list[str]) -> dict:
    """Keep only passes needed to produce ``outputs``.

    Mark every pass that writes an output resource, then repeatedly mark every
    pass that writes a resource read by an already-marked pass. Passes that
    are never marked are pruned.

    Return ``{"kept": [...], "pruned": [...]}``. Both lists contain pass names
    in the same stable topological order that :func:`toposort` defines. The
    graph must be acyclic; otherwise ``GraphError`` is raised.
    """

    if not isinstance(graph, dict):
        raise GraphError("graph is not a mapping")
    passes = graph.get("passes", [])
    if passes is None:
        passes = []
    if isinstance(passes, str) or not isinstance(passes, (list, tuple)):
        raise GraphError("passes must be a list")

    prepared = _prepare_passes(passes)
    order = _stable_topo_order(prepared)
    lookup = _pas_lookup(prepared)

    writers: dict[str, list[str]] = {}
    for pas in prepared:
        for resource in pas["writes"]:
            writers.setdefault(resource, []).append(pas["name"])

    requested = outputs or []
    kept: set[str] = set()
    stack: list[str] = []
    for resource in requested:
        for name in writers.get(resource, ()):
            if name not in kept:
                kept.add(name)
                stack.append(name)

    while stack:
        name = stack.pop()
        for resource in lookup[name]["reads"]:
            for producer in writers.get(resource, ()):
                if producer not in kept:
                    kept.add(producer)
                    stack.append(producer)

    return {
        "kept": [name for name in order if name in kept],
        "pruned": [name for name in order if name not in kept],
    }


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

    if not isinstance(graph, dict):
        return False
    passes = graph.get("passes", [])
    if passes is None:
        passes = []
    if isinstance(passes, str) or not isinstance(passes, (list, tuple)):
        return False

    try:
        prepared = _prepare_passes(passes)
        order = _stable_topo_order(prepared)
    except GraphError:
        return False

    lookup = _pas_lookup(prepared)
    if a not in lookup or b not in lookup or a == b:
        return False

    position_a = order.index(a)
    position_b = order.index(b)
    if abs(position_a - position_b) != 1:
        return False

    first, second = lookup[a], lookup[b]
    first_reads, first_writes = set(first["reads"]), set(first["writes"])
    second_reads, second_writes = set(second["reads"]), set(second["writes"])

    if first_writes & second_writes:
        return False
    if first_writes & second_reads:
        return False
    if second_writes & first_reads:
        return False

    formats = _graph_formats(graph)
    shared = (
        (first_reads | first_writes) & (second_reads | second_writes)
    )
    for resource in shared:
        format_a = formats.get((a, resource))
        format_b = formats.get((b, resource))
        if (
            format_a is not None
            and format_b is not None
            and format_a != format_b
        ):
            return False
    return True


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

    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        outputs = {}

    if graphs is None:
        graphs = []
    if isinstance(graphs, str) or not isinstance(graphs, (list, tuple)):
        return {
            "ok": False,
            "results": {},
            "errors": {"": ["graphs must be a list"]},
        }
    graphs = list(graphs)

    # --- Phase 1: validate the whole batch before executing anything. ---
    errors: dict[str, list[str]] = {}
    names: list[str] = []
    name_counts: dict[str, int] = {}

    for position, graph in enumerate(graphs):
        fallback = f"<graph {position}>"
        name = graph.get("name") if isinstance(graph, dict) else None
        key = name if isinstance(name, str) and name else fallback
        if not isinstance(name, str) or not name:
            errors.setdefault(key, []).append(
                f"graph at position {position} has no valid name"
            )
            names.append(key)
            continue
        names.append(name)
        name_counts[name] = name_counts.get(name, 0) + 1

    for name, count in name_counts.items():
        if count > 1:
            errors.setdefault(name, []).append(
                f"duplicate graph name {name!r} in batch"
            )

    for name, graph in zip(names, graphs):
        issues = validate(graph)
        if issues:
            errors.setdefault(name, []).extend(issues)

    if errors:
        # Atomic rollback: discard every partial result.
        return {"ok": False, "results": {}, "errors": errors}

    # --- Phase 2: execute; re-raise any unexpected bad data atomically. ---
    results: dict[str, dict] = {}
    failures: dict[str, list[str]] = {}
    for name, graph in zip(names, graphs):
        try:
            order = toposort(graph.get("passes") or [])
            culled = cull_unused(graph, outputs.get(name, []))
        except GraphError as exc:
            failures.setdefault(name, []).append(str(exc))
            continue
        results[name] = {
            "order": order,
            "kept": culled["kept"],
            "pruned": culled["pruned"],
        }

    if failures:
        return {"ok": False, "results": {}, "errors": failures}

    return {"ok": True, "results": results, "errors": {}}
