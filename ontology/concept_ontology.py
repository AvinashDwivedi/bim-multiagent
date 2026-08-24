"""
Loader + resolvers for ``concept_ontology.yaml`` — the Layer-2 concept ontology
(design_robust_bim_architecture.md §4; plan_robust_bim_checkpoints.md CP-2.x).

ONE declarative home for "what words/concepts map to which canonical functions",
consolidating the synonym/classification tables that drifted across the codebase:

    _GEBIEDSTYPE_TO_CANONICAL / _SPACE_LABEL_TO_CANONICAL  (chunker_bim, backfill)
    _CANONICAL_SPACE_TYPE_SYNONYMS / _CANONICAL_SPACE_FUNCTIONS  (bim_tools)
    _ELEMENT_CLASS_IFC_TYPES / _ELEMENT_CLASS_SYNONYMS          (bim_tools)
    _TERM_SYNONYMS                                              (bim_tools)
    _CONCEPT_EXPANSIONS                                         (retriever_docs_bim)

CP-2.1 ships this loader as a FAITHFUL reproduction of those tables (proved by
localdocs/check_ontology_coverage.py); it is not yet wired into the consumers
(that is CP-2.2 / CP-2.3). The resolver functions below mirror the existing Python
helpers one-for-one so a consumer can be repointed with no behaviour change:

    normalize_space_type(term)      ~ bim_tools._normalize_canonical_space_type
    resolve_space_function(term)    ~ bim_tools._resolve_canonical_space_function
    resolve_element_classes(term)   ~ bim_tools._resolve_element_ifc_classes
    gebiedstype_to_function(gt)     ~ chunker_bim._GEBIEDSTYPE_TO_CANONICAL.get(gt)
    classify_label(label)           ~ chunker_bim._canonical_space_function_from_label_with_term
    retrieval_terms_for(query)      ~ retriever_docs_bim._CONCEPT_EXPANSIONS expansion
"""

from __future__ import annotations

import copy
import os
import threading
from typing import Optional

try:
    import yaml  # PyYAML (pinned in backend/requirements.txt)
except ImportError as e:  # pragma: no cover
    raise ImportError("concept_ontology requires PyYAML (pip install pyyaml)") from e


_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "concept_ontology.yaml")
# Overlay files live alongside the global yaml, in git, diffable and reviewable
# (design_ontology_cascade.md §2 — files, not DB). A missing overlay is normal
# ("no overlay"), never an error.
_OVERLAY_DIR = os.path.join(os.path.dirname(__file__), "ontology")
_LOCK = threading.Lock()
# CP-C0: cache key is the composite (global_path, client_id, project_id) so the same
# process can hold the pure-global view and any per-client/project composed view at once.
_CACHE: dict[tuple, "ConceptOntology"] = {}


# --- CP-C0 · the Global → Client → Project merge policy (design §3) -----------------
# Priority bands keep first-match-wins ordering namespaced by layer: a lower layer can
# never reorder a higher one unless it explicitly overrides the same slug (§3.1).
_CLIENT_BAND = 1000
_PROJECT_BAND = 2000

# Per-field merge policy for a concept (§3.2). LIST-APPEND = deduped, order-preserving,
# base first (vocabulary grows across layers). Everything else is scalar-replace
# (lower layer wins) — except query_priority/classify_priority, which are band-shifted.
_CONCEPT_LIST_APPEND = frozenset({
    "aliases", "keyword_synonyms", "classify_labels", "gebiedstype",
    "retrieval_terms", "ifc_classes", "door_signals", "two_sided_markers",
})
_CONCEPT_BAND_FIELDS = ("query_priority", "classify_priority")

# permit_measures: aliases/level_markers grow; level_role replaces; every `extract`
# sub-field (block/path/unit/basis/label) replaces wholesale (bound to one JSON schema).
_MEASURE_LIST_APPEND = frozenset({"aliases", "level_markers"})


def _merge_list(base, overlay):
    """Append overlay items not already in base; dedup, order-preserving, base first."""
    out = list(base or [])
    seen = {(x if isinstance(x, (str, int, float)) else repr(x)) for x in out}
    for item in overlay or []:
        key = item if isinstance(item, (str, int, float)) else repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _band_priorities(concept: dict, band: int) -> dict:
    """Shift a NEW overlay concept's explicit priorities into its layer's band, so it
    can never sort ahead of a higher-layer concept (§3.1)."""
    out = dict(concept)
    for k in _CONCEPT_BAND_FIELDS:
        if isinstance(out.get(k), int):
            out[k] = out[k] + band
    return out


def _merge_concept(base: dict, overlay: dict, band: int) -> dict:
    """Merge an overlay concept ONTO an existing (higher-layer) concept of the same slug.
    Only keys present in the overlay are touched — so an override inherits the base slug's
    priority/band position unless it restates the priority (then it moves into the band)."""
    out = dict(base)
    for k, v in overlay.items():
        if k in _CONCEPT_LIST_APPEND:
            out[k] = _merge_list(base.get(k), v)
        elif k in _CONCEPT_BAND_FIELDS:
            out[k] = (v + band) if isinstance(v, int) else v
        else:
            out[k] = v  # scalar-replace (label, absence_note, functions, kind, …)
    return out


def _merge_measure(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in _MEASURE_LIST_APPEND:
            out[k] = _merge_list(base.get(k), v)
        elif k == "extract" and isinstance(v, dict):
            merged = dict(base.get("extract") or {})
            merged.update(v)  # each sub-field replaces
            out[k] = merged
        else:
            out[k] = v
    return out


def _merge_nested_map(base: dict, overlay: dict) -> dict:
    """Recursively merge scoped BIM query knowledge; lower layers replace values."""
    out = copy.deepcopy(base or {})
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_nested_map(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _merge_named_map(result: dict, section: str, overlay_section: dict, merge_fn):
    """Merge a top-level slug->dict section (concepts / permit_measures). New slugs add
    wholesale; existing slugs merge via merge_fn."""
    dest = dict(result.get(section) or {})
    for slug, ov in (overlay_section or {}).items():
        dest[slug] = merge_fn(dest[slug], ov) if slug in dest else ov
    result[section] = dest


def _merge_term_map(result: dict, section: str, overlay_section: dict):
    """Merge a top-level slug->list section (room_labels / materials) by list-append."""
    dest = dict(result.get(section) or {})
    for slug, terms in (overlay_section or {}).items():
        dest[slug] = _merge_list(dest.get(slug), terms) if slug in dest else list(terms or [])
    result[section] = dest


def _merge_layer(result: dict, layer: dict, band: int):
    if not layer:
        return
    # concepts — new slugs banded wholesale, existing slugs merged in place.
    dest = dict(result.get("concepts") or {})
    for slug, ov in (layer.get("concepts") or {}).items():
        dest[slug] = _merge_concept(dest[slug], ov, band) if slug in dest else _band_priorities(ov, band)
    result["concepts"] = dest
    _merge_named_map(result, "permit_measures", layer.get("permit_measures"), _merge_measure)
    _merge_term_map(result, "room_labels", layer.get("room_labels"))
    _merge_term_map(result, "materials", layer.get("materials"))
    if layer.get("bim_query_knowledge"):
        result["bim_query_knowledge"] = _merge_nested_map(
            result.get("bim_query_knowledge") or {}, layer["bim_query_knowledge"]
        )
    # `version` stays the global identity; overlays never change it.


def compose_ontology(global_data: dict,
                     client_data: Optional[dict] = None,
                     project_data: Optional[dict] = None) -> dict:
    """Compose a Global → Client → Project ontology into a single merged data dict that
    ``ConceptOntology`` consumes unchanged (design §2). With no overlays this is a faithful
    deep copy of ``global_data`` — the load-bearing empty-overlay identity invariant (§3):
    ``compose_ontology(g) == g``. Pure function; does not mutate its inputs."""
    result = copy.deepcopy(global_data) if global_data else {}
    _merge_layer(result, client_data, _CLIENT_BAND)
    _merge_layer(result, project_data, _PROJECT_BAND)
    return result


def _read_yaml(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ConceptOntology:
    """Parsed view of concept_ontology.yaml with the resolver helpers."""

    def __init__(self, data: dict):
        self.version = data.get("version")
        self.concepts: dict[str, dict] = dict(data.get("concepts") or {})
        # CP-PM4: project-level permit measures (mirror of bim_tools._MEASURE_SYNONYMS +
        # chunker_bim._PERMIT_MEASURE_EXTRACT + the two _*_measure_for_level helpers).
        self.permit_measures: dict[str, dict] = dict(data.get("permit_measures") or {})
        self.bim_query_knowledge: dict[str, dict] = dict(data.get("bim_query_knowledge") or {})
        # CP-S3: Hebrew 2D-plan room labels (slug -> exact label terms). Consumed by
        # chunker_bim._extract_room_label_nodes via classify_room_label().
        self.room_labels: dict[str, list[str]] = dict(data.get("room_labels") or {})
        self._room_label_lookup: dict[str, str] = {}              # exact lowered term -> slug
        for slug, terms in self.room_labels.items():
            for t in terms or []:
                self._room_label_lookup.setdefault(str(t).strip().lower(), slug)

        # Material vocabulary (slug -> cross-language substring terms). Consumed by
        # find_elements_by_material via material_terms_for(); a user term ("wood")
        # resolves to the slug whose terms include it, expanding to the model-language
        # vocab ("hout", …) that actually appears inside materials_json.
        self.materials: dict[str, list[str]] = {
            slug: [str(t).strip().lower() for t in (terms or [])]
            for slug, terms in (data.get("materials") or {}).items()
        }
        self._material_term_lookup: dict[str, str] = {}           # term -> slug
        for slug, terms in self.materials.items():
            self._material_term_lookup.setdefault(slug.lower(), slug)
            for t in terms:
                self._material_term_lookup.setdefault(t, slug)

        # Pre-index for the resolvers.
        self._space_funcs: list[tuple[int, str, list[str]]] = []   # (query_priority, slug, aliases)
        self._classify: list[tuple[int, str, list[str]]] = []      # (classify_priority, slug, labels)
        self._gebiedstype: dict[str, str] = {}                     # pset value -> slug
        self._element_alias: dict[str, list[str]] = {}            # alias -> ifc_classes
        self._space_func_slugs: set[str] = set()
        # keyword-expansion synonyms (mirror of bim_tools._TERM_SYNONYMS) — a SEPARATE
        # vocabulary from `aliases` (which feed the substring-matched resolvers), so an
        # over-broad alias list (e.g. the elevator/vertical_circulation overlap) can't
        # leak into grep expansion. member-term (lowered) -> the full synonym list.
        self._keyword_synonyms: dict[str, list[str]] = {}

        for slug, c in self.concepts.items():
            kind = c.get("kind")
            if kind == "space_function":
                self._space_func_slugs.add(slug)
                qp = c.get("query_priority", 10_000)
                self._space_funcs.append((qp, slug, [a.lower() for a in c.get("aliases", [])]))
                labels = c.get("classify_labels") or []
                if labels:
                    cp = c.get("classify_priority", 10_000)
                    self._classify.append((cp, slug, [str(x).lower() for x in labels]))
                for gt in c.get("gebiedstype") or []:
                    self._gebiedstype[str(gt).strip().lower()] = slug
            elif kind == "element_class":
                ifc = list(c.get("ifc_classes") or [])
                for a in c.get("aliases", []):
                    self._element_alias[str(a).lower()] = ifc

            # keyword-expansion synonyms (any kind). First-wins on member term, mirroring
            # _expand_term_synonyms' first-match-in-dict-order (the _TERM_SYNONYMS lists
            # are disjoint, so order is immaterial in practice).
            ks = c.get("keyword_synonyms")
            if ks:
                ks = list(ks)
                for member in ks:
                    self._keyword_synonyms.setdefault(str(member).lower(), ks)

        self._space_funcs.sort(key=lambda t: t[0])
        self._classify.sort(key=lambda t: t[0])

    # -- query-time space resolution (mirrors _normalize_canonical_space_type) ----
    def normalize_space_type(self, term: Optional[str]) -> str:
        tl = (term or "").lower().strip()
        if not tl:
            return tl
        for _qp, slug, aliases in self._space_funcs:
            if tl == slug or any(a in tl or tl in a for a in aliases):
                return slug
        return tl

    def resolve_space_function(self, term: Optional[str]) -> Optional[str]:
        if not term:
            return None
        norm = self.normalize_space_type(term)
        return norm if norm in self._space_func_slugs else None

    def space_function_slugs(self) -> set[str]:
        return set(self._space_func_slugs)

    def element_class_ifc_types(self) -> dict[str, list[str]]:
        """{element_class slug -> its exact IFC leaf types}, the ontology equivalent of
        bim_tools._ELEMENT_CLASS_IFC_TYPES (consumed by count anchors + gen_ground_truth)."""
        out: dict[str, list[str]] = {}
        for slug, c in self.concepts.items():
            if c.get("kind") == "element_class":
                out[slug] = list(c.get("ifc_classes") or [])
        return out

    # -- element class resolution (mirrors _resolve_element_ifc_classes) ----------
    def resolve_element_classes(self, term: Optional[str]) -> Optional[list[str]]:
        if not term:
            return None
        tl = term.lower().strip()
        ifc = self._element_alias.get(tl)
        if ifc is None and tl.endswith("s"):
            ifc = self._element_alias.get(tl[:-1])
        return list(ifc) if ifc is not None else None

    # -- ingest classification (mirror of the chunker_bim / backfill classifiers) -
    def gebiedstype_to_function(self, gt: Optional[str]) -> Optional[str]:
        if not gt:
            return None
        return self._gebiedstype.get(str(gt).strip().lower())

    def classify_label(self, label: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Ordered first-match over classify_labels (the ingest _SPACE_LABEL_TO_CANONICAL
        path). Returns (slug, matched_alias) or (None, None)."""
        tl = (label or "").lower()
        if not tl:
            return None, None
        for _cp, slug, labels in self._classify:
            for a in labels:
                if a in tl:
                    return slug, a
        return None, None

    def classify_room_label(self, text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """CP-S3: classify a 2D-plan room label (Hebrew IfcAnnotation Text Content) to a
        room slug by EXACT match of the stripped text — or its REVERSAL (DWFX stores some
        Hebrew glyphs RTL-reversed, e.g. 'הינח' = 'חניה'/parking). Exact (not substring) so
        a drawing TITLE like 'חתך מחסן' (storage section) is not read as a storage ROOM.
        Returns (slug, matched_term) or (None, None)."""
        if not text:
            return None, None
        s = str(text).strip().lower()
        slug = self._room_label_lookup.get(s)
        if slug:
            return slug, s
        r = str(text)[::-1].strip().lower()
        slug = self._room_label_lookup.get(r)
        if slug:
            return slug, r
        return None, None

    def material_terms_for(self, term: Optional[str]) -> tuple[Optional[str], list[str]]:
        """Resolve a user material term (any language) → (slug, model-vocab terms).
        Matches `term` against each material's term list by substring (either
        direction), longest-term match wins so 'metaal_aluminium' beats 'metaal'.
        Returns (slug, that material's full term list) so the caller can grep the
        model-language vocabulary inside materials_json. Falls back to
        (None, [lowered term]) when nothing matches, so the literal term is still
        tried (a model may store English material names)."""
        tl = (term or "").lower().strip()
        if not tl:
            return None, []
        best_slug, best_len = None, 0
        for t, slug in self._material_term_lookup.items():
            if (t in tl or tl in t) and len(t) > best_len:
                best_slug, best_len = slug, len(t)
        if best_slug:
            return best_slug, list(self.materials.get(best_slug, [tl]))
        return None, [tl]

    # -- L3 keyword-retrieval expansion (mirror of _CONCEPT_EXPANSIONS) -----------
    def retrieval_terms_for(self, query: Optional[str]) -> list[str]:
        """Concept-driven keyword expansion: for every concept whose alias appears in
        the query, contribute its retrieval_terms. Faithfully reproduces the
        vertical-transport expansion; generalises it to all concepts (a superset —
        harmless until CP-3.3 wires this into _search_bim_by_keyword)."""
        ql = (query or "").lower()
        if not ql:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for slug, c in self.concepts.items():
            terms = c.get("retrieval_terms") or []
            if not terms:
                continue
            triggers = [str(a).lower() for a in c.get("aliases", [])]
            if any(t in ql for t in triggers):
                for term in terms:
                    if term not in seen:
                        seen.add(term)
                        out.append(term)
        return out

    def keyword_synonyms_for(self, term: Optional[str]) -> list[str]:
        """Fuzzy keyword-expansion synonyms for a search term (mirror of
        bim_tools._expand_term_synonyms). Returns the full synonym list of the
        concept whose `keyword_synonyms` contains `term`, else ``[term]`` (the
        original-cased term) when nothing matches."""
        tl = (term or "").lower().strip()
        return self._keyword_synonyms.get(tl, [term])

    def absence_note(self, concept: str) -> Optional[str]:
        c = self.concepts.get(concept)
        return (c or {}).get("absence_note")

    def functions_of(self, concept: str) -> list[str]:
        """CP-4.1 computed relevance: the canonical_type/function slugs that SATISFY a
        concept — i.e. an element is relevant to this concept iff its canonical_function ∈
        this list. Returns ONLY an EXPLICIT `functions:` declaration (the cross-cutting case
        like vertical_circulation → [stair, circulation, lift]); a plain space_function whose
        function is just itself returns [] (nothing useful to add). Empty if absent."""
        c = self.concepts.get(concept) or {}
        return [str(x) for x in (c.get("functions") or [])]

    def absence_notes_for_query(self, query: Optional[str]) -> list[tuple[str, str]]:
        """CP-4.3: honest-absence composition. For every concept whose alias appears in
        the query AND that declares an `absence_note`, return (slug, note). Alias-driven
        and deterministic (same trigger model as `retrieval_terms_for`), single-homed in
        the yaml — so the synthesis can state the honest "this literal element is often
        not modelled; here is what IS present" wording instead of inventing or dismissing.
        Ordered by concept declaration order, deduped by slug."""
        ql = (query or "").lower()
        if not ql:
            return []
        out: list[tuple[str, str]] = []
        for slug, c in self.concepts.items():
            note = c.get("absence_note")
            if not note:
                continue
            triggers = [str(a).lower() for a in c.get("aliases", [])]
            if any(t in ql for t in triggers):
                out.append((slug, " ".join(str(note).split())))
        return out

    def retrieval_terms_of(self, concept: str) -> list[str]:
        """The `retrieval_terms` of ONE named concept (by slug), or [] if absent.
        Distinct from `retrieval_terms_for(query)` which scans all concepts by alias —
        this is the direct lookup used to source curated prose vocab (graph.py)."""
        c = self.concepts.get(concept) or {}
        return [str(x) for x in (c.get("retrieval_terms") or [])]

    def door_signals(self, concept: str) -> list[str]:
        """Lowercased landing-door markers for a concept (e.g. elevator → ['brandweerlift']).
        A door whose name/psets carry one of these IS a landing door of that concept even
        when the model has no discrete IfcTransportElement/shaft node (the F4 case)."""
        c = self.concepts.get(concept) or {}
        return [str(x).lower() for x in (c.get("door_signals") or [])]

    def two_sided_markers(self, concept: str) -> list[str]:
        """Lowercased markers that, on a landing door's Reference, mean the lift is
        accessed from two sides — so a floor's two landing doors serve ONE shaft."""
        c = self.concepts.get(concept) or {}
        return [str(x).lower() for x in (c.get("two_sided_markers") or [])]

    # -- CP-PM4 permit measures (mirror of the bim_tools / chunker_bim measure tables) --
    def permit_measure_extract(self) -> list[tuple]:
        """Ingest extraction spec, one tuple per measure that declares `extract`, in
        yaml order: (measure_key, block_id, path, unit, basis, label). Reproduces
        chunker_bim._PERMIT_MEASURE_EXTRACT (consumed by _build_canonical_measure_nodes)."""
        out: list[tuple] = []
        for key, m in self.permit_measures.items():
            ex = m.get("extract")
            if not isinstance(ex, dict):
                continue
            out.append((
                key,
                ex.get("block"),
                list(ex.get("path") or []),
                ex.get("unit"),
                ex.get("basis"),
                ex.get("label"),
            ))
        return out

    def resolve_measure_key(self, term: Optional[str]) -> Optional[str]:
        """Free-text EN/HE measure name → measure key (longest-alias-match wins).
        Reproduces bim_tools._resolve_measure_key over _MEASURE_SYNONYMS."""
        if not term:
            return None
        t = term.strip().lower()
        best_key, best_len = None, 0
        for key, m in self.permit_measures.items():
            for a in m.get("aliases") or []:
                al = str(a).lower()
                if al in t and len(al) > best_len:
                    best_key, best_len = key, len(al)
        return best_key

    def _measure_for_level(self, level_filter: Optional[str], role: str) -> Optional[str]:
        """First measure with `level_role == role` whose level_markers match the filter.
        Reproduces _floor_area_measure_for_level (role='floor_area') and
        _floor_height_measure_for_level (role='floor_height')."""
        if not level_filter:
            return None
        lv = level_filter.strip().lower()
        for key, m in self.permit_measures.items():
            if m.get("level_role") != role:
                continue
            if any(str(mk).lower() in lv for mk in m.get("level_markers") or []):
                return key
        return None

    def floor_area_measure_for_level(self, level_filter: Optional[str]) -> Optional[str]:
        return self._measure_for_level(level_filter, "floor_area")

    def floor_height_measure_for_level(self, level_filter: Optional[str]) -> Optional[str]:
        return self._measure_for_level(level_filter, "floor_height")


def load_ontology(path: Optional[str] = None, *, client_id: Optional[str] = None,
                  project_id: Optional[str] = None, reload: bool = False) -> ConceptOntology:
    """Load (and cache) the composed concept ontology (design §2 cascade). Thread-safe.

    With no ids this returns the pure Global view (byte-for-byte today's behaviour). With
    ``client_id`` / ``project_id`` it reads ``ontology/clients/<client_id>.yaml`` and
    ``ontology/projects/<project_id>.yaml`` **if present** and composes Global → Client →
    Project. A missing overlay file is normal ("no overlay"), never an error. Cached per
    ``(global_path, client_id, project_id)``; editing an overlay needs ``reload=True`` or a
    process restart."""
    p = path or _DEFAULT_PATH
    key = (p, client_id, project_id)
    if not reload and key in _CACHE:
        return _CACHE[key]
    with _LOCK:
        if reload or key not in _CACHE:
            global_data = _read_yaml(p) or {}
            client_data = (_read_yaml(os.path.join(_OVERLAY_DIR, "clients", f"{client_id}.yaml"))
                           if client_id else None)
            project_data = (_read_yaml(os.path.join(_OVERLAY_DIR, "projects", f"{project_id}.yaml"))
                            if project_id else None)
            composed = compose_ontology(global_data, client_data, project_data)
            _CACHE[key] = ConceptOntology(composed)
    return _CACHE[key]
