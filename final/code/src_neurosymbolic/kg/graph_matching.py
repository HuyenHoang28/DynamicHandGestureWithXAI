import argparse
import json
from functools import lru_cache
from pathlib import Path


CONCEPT_RELATIONS = {
    "HAS_MOTION",
    "HAS_POSITION",
    "HAS_HAND_SHAPE",
    "HAS_FINGER_MOTION",
    "HAS_COORDINATION",
    "HAS_SWEEP",
    "HAS_JOINT_ANGLE",
}
ACTION_ROLE_RELATIONS = {"HAS_MOTION", "HAS_FINGER_MOTION", "HAS_COORDINATION", "HAS_SWEEP"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def node_map(graph: dict) -> dict[str, dict]:
    return {node["id"]: node for node in graph.get("nodes", [])}


def node_name(nodes: dict[str, dict], node_id: str) -> str:
    node = nodes.get(node_id, {})
    props = node.get("properties", {})
    return props.get("name") or props.get("label_name") or node_id.split(":", 1)[-1]


def event_subjects(graph: dict) -> dict[str, str]:
    nodes = node_map(graph)
    subjects = {}
    for node in graph.get("nodes", []):
        if node.get("type") == "Event":
            props = node.get("properties", {})
            if props.get("subject"):
                subjects[node["id"]] = props["subject"]
    for edge in graph.get("edges", []):
        if edge.get("relation") != "INVOLVES":
            continue
        source = edge["source"]
        target = edge["target"]
        if source in subjects:
            subjects[source] = node_name(nodes, target)
    return subjects


def instance_facts(graph: dict) -> tuple[set[tuple[str, str, str]], dict[tuple[str, str, str], list[dict]]]:
    nodes = node_map(graph)
    subjects = event_subjects(graph)
    max_end_frame = 0
    for node in graph.get("nodes", []):
        if node.get("type") == "Event":
            max_end_frame = max(max_end_frame, int(node.get("properties", {}).get("end_frame") or 0))
    total_frames = max(1, max_end_frame + 1)
    facts = set()
    evidence = {}
    for edge in graph.get("edges", []):
        relation = edge.get("relation")
        if relation not in CONCEPT_RELATIONS:
            continue
        event_id = edge["source"]
        subject = subjects.get(event_id)
        if not subject:
            continue
        concept = node_name(nodes, edge["target"])
        fact = (subject, relation, concept)
        facts.add(fact)
        event_node = nodes.get(event_id, {})
        event_props = event_node.get("properties", {})
        duration = int(event_props.get("duration") or 1)
        confidence = edge.get("properties", {}).get("confidence")
        if confidence is None:
            confidence = event_props.get("confidence", 1.0)
        duration_ratio = max(0.0, min(1.0, duration / total_frames))
        duration_factor = 0.35 + 0.65 * duration_ratio
        strength = max(0.0, min(1.0, duration_factor * float(confidence or 0.0)))
        evidence.setdefault(fact, []).append(
            {
                "event_id": event_props.get("event_id", event_id),
                "event_node": event_id,
                "subject": subject,
                "relation": relation,
                "concept": concept,
                "confidence": confidence,
                "duration": duration,
                "strength": round(strength, 6),
                "start_frame": event_props.get("start_frame"),
                "end_frame": event_props.get("end_frame"),
            }
        )
    return facts, evidence


def template_requirements(graph: dict) -> list[dict]:
    nodes = node_map(graph)
    subjects = event_subjects(graph)
    requirements = []
    for edge in graph.get("edges", []):
        relation = edge.get("relation")
        props = edge.get("properties", {})
        level = props.get("requirement_level")
        if relation not in CONCEPT_RELATIONS or not level:
            continue
        event_id = edge["source"]
        subject = subjects.get(event_id, "any")
        event_props = nodes.get(event_id, {}).get("properties", {})
        concept = node_name(nodes, edge["target"])
        requirements.append(
            {
                "template_event": event_id,
                "subject": subject,
                "relation": relation,
                "concept": concept,
                "level": level,
                "weight": float(props.get("weight", 0.0)),
                "action_role": event_props.get("action_role", "any"),
                "fact": (subject, relation, concept),
            }
        )
    return requirements


def fact_matches(fact: tuple[str, str, str], instance_fact_set: set[tuple[str, str, str]]) -> bool:
    subject, relation, concept = fact
    if fact in instance_fact_set:
        return True
    if subject == "any":
        return any(inst_relation == relation and inst_concept == concept for _, inst_relation, inst_concept in instance_fact_set)
    return False


def evidence_for_fact(fact: tuple[str, str, str], evidence: dict[tuple[str, str, str], list[dict]]) -> list[dict]:
    subject, relation, concept = fact
    if subject != "any":
        return evidence.get(fact, [])
    matched = []
    for inst_fact, items in evidence.items():
        if inst_fact[1] == relation and inst_fact[2] == concept:
            matched.extend(items)
    return matched


def fact_strength(fact: tuple[str, str, str], evidence: dict[tuple[str, str, str], list[dict]]) -> float:
    items = evidence_for_fact(fact, evidence)
    if not items:
        return 0.0
    return max(float(item.get("strength") or 0.0) for item in items)


def instance_event_records(graph: dict) -> list[dict]:
    """Return temporally ordered event-local facts without collapsing the clip to a set."""
    nodes = node_map(graph)
    subjects = event_subjects(graph)
    max_end_frame = 0
    for node in graph.get("nodes", []):
        if node.get("type") == "Event":
            max_end_frame = max(max_end_frame, int(node.get("properties", {}).get("end_frame") or 0))
    total_frames = max(1, max_end_frame + 1)
    records = {}
    for node in graph.get("nodes", []):
        if node.get("type") != "Event":
            continue
        props = node.get("properties", {})
        records[node["id"]] = {
            "event_id": node["id"],
            "subject": subjects.get(node["id"], props.get("subject", "")),
            "start_frame": int(props.get("start_frame") or 0),
            "end_frame": int(props.get("end_frame") or props.get("start_frame") or 0),
            "confidence": float(props.get("confidence", 1.0)),
            "facts": set(),
            "fact_occurrence_ids": {},
            "fact_action_roles": {},
            "fact_strengths": {},
            "atomic_event_ids": tuple(sorted(set(props.get("atomic_event_ids") or []))),
            "event_ids": [node["id"]],
        }
    for edge in graph.get("edges", []):
        relation = edge.get("relation")
        record = records.get(edge.get("source"))
        if record is None or relation not in CONCEPT_RELATIONS:
            continue
        concept = node_name(nodes, edge.get("target"))
        fact = (record["subject"], relation, concept)
        record["facts"].add(fact)
        atomic_event_ids = (
            edge.get("properties", {}).get("atomic_event_ids")
            or record["atomic_event_ids"]
        )
        occurrence_ids = set(record["fact_occurrence_ids"].get(fact, ()))
        occurrence_ids.update(atomic_event_ids or ())
        record["fact_occurrence_ids"][fact] = tuple(sorted(occurrence_ids))
        roles = edge.get("properties", {}).get("action_roles")
        if roles:
            record["fact_action_roles"][fact] = tuple(sorted(set(roles)))

        event_node = nodes.get(record["event_id"], {})
        event_props = event_node.get("properties", {})
        duration = int(event_props.get("duration") or 1)
        confidence = edge.get("properties", {}).get("confidence")
        if confidence is None:
            confidence = event_props.get("confidence", 1.0)
        duration_ratio = max(0.0, min(1.0, duration / total_frames))
        duration_factor = 0.35 + 0.65 * duration_ratio
        strength = max(0.0, min(1.0, duration_factor * float(confidence or 0.0)))
        record["fact_strengths"][fact] = max(
            float(record["fact_strengths"].get(fact, 0.0)), strength
        )
    return sorted(
        records.values(),
        key=lambda item: (item["start_frame"], item["end_frame"], item["event_id"]),
    )


def template_event_groups(graph: dict) -> list[dict]:
    nodes = node_map(graph)
    subjects = event_subjects(graph)
    groups = {}
    for node in graph.get("nodes", []):
        if node.get("type") != "Event":
            continue
        props = node.get("properties", {})
        if props.get("negative"):
            continue
        groups[node["id"]] = {
            "event_id": node["id"],
            "subject": subjects.get(node["id"], props.get("subject", "any")),
            "ordinal": int(props.get("ordinal") or 0),
            "min_occurrences": max(1, int(props.get("min_occurrences") or 1)),
            "max_occurrences": (
                int(props["max_occurrences"])
                if props.get("max_occurrences") is not None
                else None
            ),
            "requirements": [],
            "action_role": props.get("action_role", "any"),
        }
    for requirement in template_requirements(graph):
        group = groups.get(requirement["template_event"])
        if group is not None and requirement["level"] != "negative":
            group["requirements"].append(requirement)
    return sorted(groups.values(), key=lambda item: (item["ordinal"], item["event_id"]))


def event_satisfies_template(instance_event: dict, template_event: dict) -> bool:
    required = [req for req in template_event["requirements"] if req["level"] == "required"]
    if not required:
        return False
    facts = instance_event["facts"]
    for requirement in required:
        subject, relation, concept = requirement["fact"]
        if subject != "any" and subject != instance_event["subject"]:
            return False
        if not any(
            fact_relation == relation and fact_concept == concept
            for _, fact_relation, fact_concept in facts
        ):
            return False
        if (
            template_event.get("action_role", "any") != "any"
            and relation in ACTION_ROLE_RELATIONS
            and requirement.get("action_role", "any") != "any"
        ):
            roles = instance_event.get("fact_action_roles", {}).get(
                (instance_event["subject"], relation, concept), ()
            )
            if template_event["action_role"] not in roles:
                return False
    return True


def event_occurrence_key(instance_event: dict, template_event: dict) -> tuple[str, ...]:
    """Identify the atomic evidence behind one matched gesture occurrence.

    Phase construction may split one long atomic predicate into several phase
    events.  Counting phase nodes would turn that single gesture into a false
    double-click, so occurrence identity is derived only from atomic evidence
    for the template's required facts.
    """

    occurrence_ids = set()
    for requirement in template_event["requirements"]:
        if requirement["level"] != "required":
            continue
        _, relation, concept = requirement["fact"]
        for fact, atomic_ids in instance_event.get("fact_occurrence_ids", {}).items():
            if fact[1] == relation and fact[2] == concept:
                occurrence_ids.update(atomic_ids)
    if occurrence_ids:
        return tuple(sorted(occurrence_ids))
    return (str(instance_event["event_id"]),)


def _record_atomic_ids(instance_event: dict) -> set[str]:
    """Return all atomic evidence IDs attached to an event or episode."""
    atomic_ids = set(instance_event.get("atomic_event_ids") or ())
    for occurrence_ids in instance_event.get("fact_occurrence_ids", {}).values():
        atomic_ids.update(occurrence_ids or ())
    return {str(item) for item in atomic_ids if item is not None}


def _record_matches_requirement(instance_event: dict, requirement: dict) -> bool:
    """Check one requirement against one subject-local event record."""
    subject, relation, concept = requirement["fact"]
    if subject != "any" and subject != instance_event["subject"]:
        return False
    fact_match = any(
        fact_relation == relation and fact_concept == concept
        for _, fact_relation, fact_concept in instance_event.get("facts", ())
    )
    if not fact_match:
        return False
    role = requirement.get("action_role", "any")
    if role == "any" or relation not in ACTION_ROLE_RELATIONS:
        return True
    return role in instance_event.get("fact_action_roles", {}).get(
        (instance_event["subject"], relation, concept), ()
    )


def _record_matching_facts(instance_event: dict, requirement: dict) -> list[tuple[str, str, str]]:
    """Return the actual facts satisfying a requirement in one record."""
    subject, relation, concept = requirement["fact"]
    if subject != "any" and subject != instance_event["subject"]:
        return []
    return [
        fact
        for fact in instance_event.get("facts", ())
        if fact[1] == relation and fact[2] == concept
    ]


OCCURRENCE_ANCHOR_RELATIONS = {
    "HAS_MOTION",
    "HAS_FINGER_MOTION",
    "HAS_COORDINATION",
    "HAS_SWEEP",
}


def _occurrence_anchor_requirements(template_event: dict) -> list[dict]:
    """Choose facts that identify repeated actions rather than persistent pose.

    A hand shape can span multiple real clicks.  When a template has required
    motion evidence, occurrence identity must therefore come from that motion;
    otherwise one long-lived shape atomic ID can collapse distinct actions.
    """
    positive = [
        requirement
        for requirement in template_event["requirements"]
        if requirement["level"] != "negative"
    ]
    required_motion = [
        requirement
        for requirement in positive
        if requirement["level"] == "required"
        and requirement["relation"] in OCCURRENCE_ANCHOR_RELATIONS
    ]
    if required_motion:
        return required_motion
    motion = [
        requirement
        for requirement in positive
        if requirement["relation"] in OCCURRENCE_ANCHOR_RELATIONS
    ]
    return motion or positive


def _record_occurrence_ids(instance_event: dict, template_event: dict) -> set[str]:
    """Return atomic IDs for the template's occurrence-defining evidence."""
    occurrence_ids = set()
    for requirement in _occurrence_anchor_requirements(template_event):
        for fact in _record_matching_facts(instance_event, requirement):
            occurrence_ids.update(
                instance_event.get("fact_occurrence_ids", {}).get(fact, ())
            )
    if occurrence_ids:
        return {str(item) for item in occurrence_ids if item is not None}

    # Legacy graphs may only carry IDs on the composite event.
    return {
        str(item)
        for item in instance_event.get("atomic_event_ids", ())
        if item is not None
    }


def _records_share_matching_fact(
    first: dict,
    second: dict,
    template_event: dict,
) -> bool:
    """Whether two nearby records carry the same template-relevant evidence."""
    required_keys = {
        (req["relation"], req["concept"])
        for req in _occurrence_anchor_requirements(template_event)
    }
    if not required_keys:
        return False
    first_keys = {(fact[1], fact[2]) for fact in first.get("facts", ())}
    second_keys = {(fact[1], fact[2]) for fact in second.get("facts", ())}
    return bool(required_keys & first_keys & second_keys)


def _records_are_nearby(first: dict, second: dict) -> bool:
    """Treat overlapping or immediately adjacent phase records as nearby."""
    return (
        first["start_frame"] <= second["end_frame"] + 1
        and second["start_frame"] <= first["end_frame"] + 1
    )


def _merge_event_records(first: dict, second: dict) -> dict:
    """Merge duplicate phase fragments into one episode without losing evidence."""
    merged = dict(first)
    merged["facts"] = set(first.get("facts", ())) | set(second.get("facts", ()))
    merged["fact_occurrence_ids"] = {
        fact: tuple(
            sorted(
                set(first.get("fact_occurrence_ids", {}).get(fact, ()))
                | set(second.get("fact_occurrence_ids", {}).get(fact, ()))
            )
        )
        for fact in set(first.get("fact_occurrence_ids", {}))
        | set(second.get("fact_occurrence_ids", {}))
    }
    merged["fact_strengths"] = {
        fact: max(
            float(first.get("fact_strengths", {}).get(fact, 0.0)),
            float(second.get("fact_strengths", {}).get(fact, 0.0)),
        )
        for fact in set(first.get("fact_strengths", {}))
        | set(second.get("fact_strengths", {}))
    }
    merged["fact_action_roles"] = {
        fact: tuple(sorted(set(first.get("fact_action_roles", {}).get(fact, ()))
                           | set(second.get("fact_action_roles", {}).get(fact, ()))))
        for fact in set(first.get("fact_action_roles", {})) | set(second.get("fact_action_roles", {}))
    }
    merged["atomic_event_ids"] = tuple(
        sorted(_record_atomic_ids(first) | _record_atomic_ids(second))
    )
    merged["event_ids"] = list(
        dict.fromkeys(
            list(first.get("event_ids", [first["event_id"]]))
            + list(second.get("event_ids", [second["event_id"]]))
        )
    )
    merged["start_frame"] = min(first["start_frame"], second["start_frame"])
    merged["end_frame"] = max(first["end_frame"], second["end_frame"])
    merged["duration"] = merged["end_frame"] - merged["start_frame"] + 1
    merged["confidence"] = max(float(first.get("confidence", 0.0)), float(second.get("confidence", 0.0)))
    return merged


def deduplicate_occurrence_matches(matches: list[dict], template_event: dict) -> list[dict]:
    """Collapse duplicate/overlapping fragments while retaining separate gestures.

    Atomic IDs are the strongest identity signal.  For older graphs without those
    IDs, only records that are overlapping/adjacent *and* share a template-relevant
    fact are collapsed.  A gap therefore remains sufficient to distinguish two
    real occurrences, while phase fragments cannot manufacture a double click.
    """
    episodes: list[dict] = []
    for event in sorted(
        matches,
        key=lambda item: (item["start_frame"], item["end_frame"], item["event_id"]),
    ):
        event_occurrence_ids = _record_occurrence_ids(event, template_event)
        merged = False
        for index, episode in enumerate(episodes):
            episode_occurrence_ids = _record_occurrence_ids(episode, template_event)
            identities_available = bool(event_occurrence_ids and episode_occurrence_ids)
            same_atomic_evidence = bool(event_occurrence_ids & episode_occurrence_ids)
            nearby_duplicate = (
                not identities_available
                and _records_are_nearby(episode, event)
                and _records_share_matching_fact(episode, event, template_event)
            )
            if same_atomic_evidence or nearby_duplicate:
                episodes[index] = _merge_event_records(episode, event)
                merged = True
                break
        if not merged:
            episode = dict(event)
            episode["event_ids"] = list(event.get("event_ids", [event["event_id"]]))
            episode["facts"] = set(event.get("facts", ()))
            episode["fact_occurrence_ids"] = {
                fact: tuple(ids)
                for fact, ids in event.get("fact_occurrence_ids", {}).items()
            }
            episode["fact_strengths"] = dict(event.get("fact_strengths", {}))
            episodes.append(episode)
    return episodes


def template_event_episodes(instance_events: list[dict], template_event: dict) -> list[dict]:
    """Build candidate same-subject episodes for one template event."""
    candidates_by_subject: dict[str, list[dict]] = {}
    for event in instance_events:
        if template_event["subject"] != "any" and event["subject"] != template_event["subject"]:
            continue
        if not any(
            _record_matches_requirement(event, requirement)
            for requirement in template_event["requirements"]
            if requirement["level"] != "negative"
        ):
            continue
        candidates_by_subject.setdefault(event["subject"], []).append(event)

    episodes = []
    for subject_events in candidates_by_subject.values():
        episodes.extend(deduplicate_occurrence_matches(subject_events, template_event))
    return sorted(
        episodes,
        key=lambda item: (item["start_frame"], item["end_frame"], item["event_id"]),
    )


def ordered_sequence_match(
    instance_events: list[dict],
    template_events: list[dict],
) -> tuple[int, str | None, list[dict]]:
    """Match an ordered template without reusing or overlapping evidence.

    The previous V2 sequence scan only advanced an index in the phase-event
    list.  Adjacent phase nodes can overlap and can carry the same atomic
    predicate IDs, so one physical action could satisfy multiple template
    steps.  For temporal gestures (notably G07), require strictly non-overlap
    and disjoint occurrence-defining atomic evidence on one subject.

    Candidates are considered by earliest end frame, with bounded-depth
    backtracking across template phases. Backtracking matters when the earliest
    interval shares atomic evidence needed by a later step but another valid
    interval does not.
    """

    candidate_subjects = sorted({event["subject"] for event in instance_events})
    best_matches: list[dict] = []
    best_subject = None
    for subject in candidate_subjects:
        candidate_pools = []
        for template_event in template_events:
            candidates = [
                event
                for event in template_event_episodes(instance_events, template_event)
                if event["subject"] == subject
                and event_satisfies_template(event, template_event)
            ]
            candidates.sort(
                key=lambda item: (
                    int(item["end_frame"]),
                    int(item["start_frame"]),
                    str(item["event_id"]),
                )
            )
            candidate_pools.append(
                [
                    (candidate, frozenset(_record_occurrence_ids(candidate, template_event)))
                    for candidate in candidates
                ]
            )

        @lru_cache(maxsize=None)
        def search(
            template_index: int,
            previous_end: int,
            used_occurrence_ids: frozenset[str],
        ) -> tuple[int, ...]:
            if template_index >= len(template_events):
                return ()
            best: tuple[int, ...] = ()
            remaining = len(template_events) - template_index
            for candidate_index, (candidate, occurrence_ids) in enumerate(
                candidate_pools[template_index]
            ):
                if int(candidate["start_frame"]) <= previous_end:
                    continue
                if occurrence_ids and occurrence_ids & used_occurrence_ids:
                    continue
                tail = search(
                    template_index + 1,
                    int(candidate["end_frame"]),
                    used_occurrence_ids | occurrence_ids,
                )
                current = (candidate_index,) + tail
                if len(current) > len(best):
                    best = current
                if len(best) == remaining:
                    break
            return best

        selected_indices = search(0, -1, frozenset())
        subject_matches = []
        for template_index, candidate_index in enumerate(selected_indices):
            template_event = template_events[template_index]
            selected, selected_ids = candidate_pools[template_index][candidate_index]
            subject_matches.append(
                {
                    "template_event": template_event["event_id"],
                    "instance_events": list(
                        selected.get("event_ids", [selected["event_id"]])
                    ),
                    "start_frame": int(selected["start_frame"]),
                    "end_frame": int(selected["end_frame"]),
                    "occurrence_ids": sorted(selected_ids),
                }
            )
        if len(subject_matches) > len(best_matches):
            best_matches = subject_matches
            best_subject = subject
    return len(best_matches), best_subject, best_matches


def _record_requirement_strength(instance_event: dict, requirement: dict) -> float:
    matching_facts = _record_matching_facts(instance_event, requirement)
    if not matching_facts:
        return 0.0
    strengths = instance_event.get("fact_strengths", {})
    return max(float(strengths.get(fact, 0.0)) for fact in matching_facts)


def _record_evidence_for_requirement(
    instance_event: dict,
    requirement: dict,
    evidence: dict[tuple[str, str, str], list[dict]],
) -> list[dict]:
    event_nodes = set(instance_event.get("event_ids", [instance_event["event_id"]]))
    matching = []
    for fact in _record_matching_facts(instance_event, requirement):
        for item in evidence.get(fact, []):
            if item.get("event_node") in event_nodes:
                matching.append(item)
    return matching


def _best_template_event_candidate(
    candidates: list[dict],
    template_event: dict,
) -> tuple[dict | None, list[dict], list[dict]]:
    """Select one candidate episode and return its matched/missing requirements."""
    positive_requirements = [
        req for req in template_event["requirements"] if req["weight"] > 0
    ]
    required_requirements = [
        req for req in positive_requirements if req["level"] == "required"
    ]

    def candidate_key(candidate: dict) -> tuple[float, float, float, int, int]:
        matched_required = sum(
            1 for req in required_requirements if _record_matches_requirement(candidate, req)
        )
        required_strength = sum(
            _record_requirement_strength(candidate, req) for req in required_requirements
        )
        positive_score = sum(
            req["weight"] * _record_requirement_strength(candidate, req)
            for req in positive_requirements
        )
        return (
            float(matched_required),
            required_strength,
            positive_score,
            -int(candidate["start_frame"]),
            -int(candidate["end_frame"]),
        )

    best = max(candidates, key=candidate_key, default=None)
    if best is None:
        return None, [], required_requirements
    matched = [
        req for req in positive_requirements if _record_matches_requirement(best, req)
    ]
    missing_required = [
        req for req in required_requirements if not _record_matches_requirement(best, req)
    ]
    return best, matched, missing_required


def _candidate_role_diagnostics(
    candidate: dict,
    event_requirements: list[dict],
) -> set[str]:
    """Return shared roles for a complete required directional conjunction.

    Diagnostics ignore the template's hard role filter. This lets templates
    keep ``action_role: any`` for recall while arbitration inspects every
    qualifying active/return episode independently.
    """
    required = [req for req in event_requirements if req["level"] == "required"]
    role_agnostic = [{**req, "action_role": "any"} for req in required]
    if not required or not all(
        _record_matches_requirement(candidate, req) for req in role_agnostic
    ):
        return set()
    directional = [
        req for req in role_agnostic if req["fact"][1] in ACTION_ROLE_RELATIONS
    ]
    if not directional:
        return set()
    role_sets = []
    for req in directional:
        roles = set()
        for fact in _record_matching_facts(candidate, req):
            roles.update(candidate.get("fact_action_roles", {}).get(fact, ()))
        if not roles:
            return {"unknown"}
        role_sets.append(roles)
    shared = set.intersection(*role_sets)
    return shared or {"ambiguous"}


def temporal_template_score(instance_graph: dict, template_graph: dict) -> dict:
    template_events = template_event_groups(template_graph)
    instance_events = instance_event_records(instance_graph)
    schema_version = int(template_graph.get("schema_version", 1))
    if not template_events:
        return {"score": 1.0, "matched_events": 0, "required_events": 0, "event_matches": []}

    event_matches = []
    occurrence_scores = []
    max_occurrence_violation = False
    max_occurrence_penalties = []
    for template_event in template_events:
        candidate_episodes = (
            template_event_episodes(instance_events, template_event)
            if schema_version >= 2
            else instance_events
        )
        matches = [
            event
            for event in candidate_episodes
            if event_satisfies_template(event, template_event)
        ]
        minimum = template_event["min_occurrences"]
        maximum = template_event["max_occurrences"]
        matches_by_subject = {}
        for event in matches:
            matches_by_subject.setdefault(event["subject"], []).append(event)
        subject_scores = []
        for subject, subject_matches in matches_by_subject.items():
            if schema_version >= 2:
                unique_matches = deduplicate_occurrence_matches(
                    subject_matches,
                    template_event,
                )
            else:
                unique_matches = list(
                    {
                        event_occurrence_key(event, template_event): event
                        for event in subject_matches
                    }.values()
                )
            count = len(unique_matches)
            minimum_score = min(1.0, count / minimum) if count else 0.0
            max_penalty = 1.0
            if maximum is not None and count > maximum:
                # Excess evidence should lower confidence without erasing all
                # useful evidence.  A max/count ratio is bounded and monotonic.
                max_penalty = maximum / count if count else 1.0
            subject_score = minimum_score * max_penalty
            subject_scores.append((subject_score, count, subject, unique_matches, max_penalty))
        best_score, count, matched_subject, best_matches, best_penalty = max(
            subject_scores,
            default=(0.0, 0, None, [], 1.0),
            key=lambda item: (item[0], item[1], item[2] or ""),
        )
        score = best_score
        if maximum is not None and count > maximum:
            max_occurrence_violation = True
        max_occurrence_penalties.append(best_penalty)
        occurrence_scores.append(score)
        event_matches.append(
            {
                "template_event": template_event["event_id"],
                "min_occurrences": minimum,
                "max_occurrences": maximum,
                "observed_occurrences": count,
                "matched_subject": matched_subject,
                "matched_instance_events": [
                    event_id
                    for event in best_matches
                    for event_id in event.get("event_ids", [event["event_id"]])
                ],
                "max_occurrence_penalty": round(best_penalty, 6),
            }
        )

    sequence_ratio = 1.0
    sequence_subject = None
    sequence_matches = []
    if len(template_events) > 1:
        best_matched, sequence_subject, sequence_matches = ordered_sequence_match(
            instance_events,
            template_events,
        )
        sequence_ratio = best_matched / len(template_events)

    occurrence_score = sum(occurrence_scores) / len(occurrence_scores)
    # V1's temporal result is kept for compatibility with the legacy matcher;
    # V2 uses the bounded soft penalty above instead of a catastrophic zero.
    if schema_version < 2 and max_occurrence_violation:
        score = 0.0
    else:
        score = occurrence_score * sequence_ratio
    return {
        "score": score,
        "matched_events": round(sequence_ratio * len(template_events)),
        "required_events": len(template_events),
        "occurrence_score": occurrence_score,
        "sequence_ratio": sequence_ratio,
        "max_occurrence_violation": max_occurrence_violation,
        "max_occurrence_penalty": min(max_occurrence_penalties, default=1.0),
        "event_matches": event_matches,
        "sequence_subject": sequence_subject,
        "sequence_matches": sequence_matches,
    }


def _score_template_v2(
    instance_graph: dict,
    template_graph: dict,
    requirements: list[dict],
    evidence: dict[tuple[str, str, str], list[dict]],
) -> dict:
    """Score positive V2 requirements against one best local event/episode.

    V1 intentionally scores facts over the whole clip.  V2 templates describe
    event-local conjunctions, so each template event gets at most one subject-local
    candidate for its positive evidence.  This prevents a shape on one event/hand
    and a motion on another from becoming a false full match.
    """
    instance_events = instance_event_records(instance_graph)
    template_events = template_event_groups(template_graph)
    positive_total = sum(req["weight"] for req in requirements if req["weight"] > 0)
    positive_score = 0.0
    negative_penalty = 0.0
    matched = []
    missing_required = []
    negative_hits = []
    role_intervals = {"active": [], "return": [], "ambiguous": [], "unknown": []}

    for template_event in template_events:
        event_requirements = [
            req for req in requirements if req["template_event"] == template_event["event_id"]
        ]
        candidates = template_event_episodes(instance_events, template_event)
        seen_role_intervals = set()
        for candidate in candidates:
            interval = {
                "template_event": template_event["event_id"],
                "subject": candidate["subject"],
                "start_frame": int(candidate["start_frame"]),
                "end_frame": int(candidate["end_frame"]),
                "instance_events": list(candidate.get("event_ids", [candidate["event_id"]])),
                "atomic_event_ids": sorted(_record_atomic_ids(candidate)),
            }
            interval_key = (
                interval["template_event"], interval["subject"],
                interval["start_frame"], interval["end_frame"],
                tuple(interval["instance_events"]), tuple(interval["atomic_event_ids"]),
            )
            for role in _candidate_role_diagnostics(candidate, event_requirements):
                marker = (role, interval_key)
                if role in role_intervals and marker not in seen_role_intervals:
                    role_intervals[role].append(interval)
                    seen_role_intervals.add(marker)
        best, event_matched, event_missing_required = _best_template_event_candidate(
            candidates,
            template_event,
        )
        if best is None:
            event_missing_required = [
                req for req in event_requirements if req["level"] == "required"
            ]
        for req in event_matched:
            strength = _record_requirement_strength(best, req)
            positive_score += req["weight"] * strength
            matched.append(
                {
                    **req,
                    "match_strength": round(strength, 6),
                    "evidence": _record_evidence_for_requirement(best, req, evidence),
                }
            )
        for req in event_missing_required:
            missing_required.append(req)

    facts, _ = instance_facts(instance_graph)
    for req in requirements:
        if req["weight"] >= 0:
            continue
        if not fact_matches(req["fact"], facts):
            continue
        strength = fact_strength(req["fact"], evidence)
        negative_penalty += abs(req["weight"]) * strength
        negative_hits.append(
            {
                **req,
                "match_strength": round(strength, 6),
                "evidence": evidence_for_fact(req["fact"], evidence),
            }
        )

    required_total = sum(1 for req in requirements if req["level"] == "required")
    required_hit = required_total - len(missing_required)
    required_ratio = (required_hit / required_total) if required_total else 1.0
    raw_score = positive_score - negative_penalty
    normalized_score = raw_score / positive_total if positive_total > 0 else 0.0
    normalized_score = max(0.0, min(1.0, normalized_score)) * required_ratio
    temporal = temporal_template_score(instance_graph, template_graph)
    normalized_score *= float(temporal["score"])

    return {
        "label_id": template_graph.get("label_id"),
        "label_name": template_graph.get("label_name"),
        "score": round(normalized_score, 6),
        "raw_score": round(raw_score, 6),
        "positive_score": round(positive_score, 6),
        "negative_penalty": round(negative_penalty, 6),
        "positive_total": round(positive_total, 6),
        "required_hit": required_hit,
        "required_total": required_total,
        "matched_count": len(matched),
        "missing_required_count": len(missing_required),
        "negative_hit_count": len(negative_hits),
        "matched": serialize_requirements(matched),
        "missing_required": serialize_requirements(missing_required),
        "negative_hits": serialize_requirements(negative_hits),
        "role_intervals": role_intervals,
        "active_intervals": role_intervals["active"],
        "temporal_score": round(float(temporal["score"]), 6),
        "matched_events": temporal["matched_events"],
        "required_events": temporal["required_events"],
        "event_matches": temporal["event_matches"],
        "sequence_subject": temporal.get("sequence_subject"),
        "sequence_matches": temporal.get("sequence_matches", []),
    }


def score_template(instance_graph: dict, template_graph: dict) -> dict:
    facts, evidence = instance_facts(instance_graph)
    requirements = template_requirements(template_graph)

    if int(template_graph.get("schema_version", 1)) >= 2:
        return _score_template_v2(instance_graph, template_graph, requirements, evidence)

    positive_total = sum(req["weight"] for req in requirements if req["weight"] > 0)
    positive_score = 0.0
    negative_penalty = 0.0
    matched = []
    missing_required = []
    negative_hits = []

    for req in requirements:
        hit = fact_matches(req["fact"], facts)
        if hit and req["weight"] > 0:
            strength = fact_strength(req["fact"], evidence)
            positive_score += req["weight"] * strength
            matched.append({**req, "match_strength": round(strength, 6), "evidence": evidence_for_fact(req["fact"], evidence)})
        elif hit and req["weight"] < 0:
            strength = fact_strength(req["fact"], evidence)
            negative_penalty += abs(req["weight"]) * strength
            negative_hits.append({**req, "match_strength": round(strength, 6), "evidence": evidence_for_fact(req["fact"], evidence)})
        elif req["level"] == "required":
            missing_required.append(req)

    required_total = sum(1 for req in requirements if req["level"] == "required")
    required_hit = required_total - len(missing_required)
    required_ratio = (required_hit / required_total) if required_total else 1.0
    raw_score = positive_score - negative_penalty
    normalized_score = raw_score / positive_total if positive_total > 0 else 0.0
    normalized_score = max(0.0, min(1.0, normalized_score)) * required_ratio
    temporal = temporal_template_score(instance_graph, template_graph)
    if int(template_graph.get("schema_version", 1)) >= 2:
        normalized_score *= float(temporal["score"])

    return {
        "label_id": template_graph.get("label_id"),
        "label_name": template_graph.get("label_name"),
        "score": round(normalized_score, 6),
        "raw_score": round(raw_score, 6),
        "positive_score": round(positive_score, 6),
        "negative_penalty": round(negative_penalty, 6),
        "positive_total": round(positive_total, 6),
        "required_hit": required_hit,
        "required_total": required_total,
        "matched_count": len(matched),
        "missing_required_count": len(missing_required),
        "negative_hit_count": len(negative_hits),
        "matched": serialize_requirements(matched),
        "missing_required": serialize_requirements(missing_required),
        "negative_hits": serialize_requirements(negative_hits),
        "temporal_score": round(float(temporal["score"]), 6),
        "matched_events": temporal["matched_events"],
        "required_events": temporal["required_events"],
        "event_matches": temporal["event_matches"],
    }


def serialize_requirements(requirements: list[dict]) -> list[dict]:
    serialized = []
    for req in requirements:
        item = {
            "subject": req["subject"],
            "relation": req["relation"],
            "concept": req["concept"],
            "level": req["level"],
            "weight": req["weight"],
            "template_event": req["template_event"],
        }
        if "match_strength" in req:
            item["match_strength"] = req["match_strength"]
        if "evidence" in req:
            item["evidence"] = req["evidence"]
        serialized.append(item)
    return serialized


def load_template_graphs(template_dir: Path) -> list[dict]:
    graphs = []
    for path in sorted(template_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        graph = load_json(path)
        if graph.get("graph_type") == "template_graph":
            graphs.append(graph)
    return sorted(graphs, key=lambda graph: graph.get("label_id", 10**9))


def match_instance_to_templates(instance_graph: dict, template_graphs: list[dict], top_k: int = 5) -> dict:
    scores = [score_template(instance_graph, template) for template in template_graphs]
    scores.sort(key=lambda item: (item["score"], item["required_hit"], item["positive_score"]), reverse=True)
    return {
        "video_id": instance_graph.get("video_id", ""),
        "source_path": instance_graph.get("source_path", ""),
        "prediction": scores[0]["label_name"] if scores else None,
        "prediction_label_id": scores[0]["label_id"] if scores else None,
        "top_k": scores[:top_k],
        "scores": [
            {
                "label_id": item["label_id"],
                "label_name": item["label_name"],
                "score": item["score"],
                "raw_score": item["raw_score"],
                "required_hit": item["required_hit"],
                "required_total": item["required_total"],
                "missing_required_count": item["missing_required_count"],
                "negative_hit_count": item["negative_hit_count"],
            }
            for item in scores
        ],
    }


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an instance KG against gesture template KGs.")
    parser.add_argument("--instance-graph", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("data/template_graphs"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance_graph = load_json(args.instance_graph)
    template_graphs = load_template_graphs(args.template_dir)
    result = match_instance_to_templates(instance_graph, template_graphs, args.top_k)
    if args.output:
        write_json(result, args.output, args.pretty)
    print(f"Video: {result['video_id']}")
    print(f"Prediction: {result['prediction']} ({result['prediction_label_id']})")
    print("Top scores:")
    for item in result["top_k"]:
        print(
            f"  {item['label_id']:2d} {item['label_name']:<20} "
            f"score={item['score']:.4f} required={item['required_hit']}/{item['required_total']} "
            f"missing={item['missing_required_count']} negative={item['negative_hit_count']}"
        )


if __name__ == "__main__":
    main()
