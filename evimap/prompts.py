PROFILE_SYSTEM = """You profile an unlabeled document corpus for an evidence-grounded topic map.

Return STRICT JSON:
{
  "corpus_description": "...",
  "likely_users": ["..."],
  "useful_browsing_axes": ["..."],
  "common_boilerplate_to_skip": ["..."],
  "extraction_caveats": ["..."]
}

Keep the profile lightweight. It guides evidence extraction only; it is not the final taxonomy."""


EXTRACTION_SYSTEM = """You extract evidence phrases for an evidence-grounded topic map.

Read one document and return short exact substrings from the document. These phrases are the atomic evidence units used later for grouping and span highlighting.

Extract phrases that name substantive entities, skills, requirements, actions, methods, qualities, restrictions, outcomes, sentiments, places, times, quantities, or relationships.

Rules:
- The phrase text must be an exact substring of the document.
- Prefer reusable phrases over long clauses.
- Split coordinated lists into separate phrases.
- Do not invent normalized labels or final categories.
- Skip boilerplate, navigation text, and formatting-only strings.
- If a document has only a few substantive phrases, return only those.

Return STRICT JSON:
{
  "doc_summary": "one sentence",
  "phrases": [
    {
      "text": "exact substring",
      "role_hint": "subject_or_entity|attribute_or_value|action_or_relation|condition_or_constraint|measure_or_quantity|time_or_place|reference_or_source|opinion_or_assessment|unclear",
      "axis_hints": ["lowercase_axis_hint"],
      "context_note": "short note when ambiguous"
    }
  ]
}"""


FINE_GROUP_SYSTEM = """You group evidence phrases into fine-grained topics.

Given a numbered list of evidence phrases, put phrases together only when they express the same concrete topic or near-synonymous evidence concept. Related but distinct concepts should stay separate.

Return STRICT JSON:
{"groups": [{"label": "short topic label", "members": [0, 3, 5]}]}

Use only shown integer ids. Each id can appear in at most one group. Leave singletons out."""


MID_GROUP_SYSTEM = """You group fine-grained topics into mid-level browsing groups.

Given a numbered list of leaf topics, put topics together when a reader would expect them under the same mid-level catalogue group. Be conservative: sibling subjects under a very broad theme should stay separate.

Return STRICT JSON:
{"groups": [{"label": "short group label", "members": [0, 3, 5]}]}

Use only shown integer ids. Each id can appear in at most one group. Leave singletons out."""


TOP_GROUP_SYSTEM = """You group mid-level groups into broad top-level aspects.

Given a numbered list of mid-level groups, put groups together when they share a broad top-level theme. The goal is a compact set of top headings, but genuinely different themes must stay apart.

Return STRICT JSON:
{"groups": [{"label": "short aspect label", "members": [0, 3, 5]}]}

Use only shown integer ids. Each id can appear in at most one group. Leave singletons out."""


NAME_TOPIC_SYSTEM = """Name one fine-grained evidence topic.

Given member evidence phrases and a few supporting documents, return a concise label grounded in the phrases. Do not add facts not supported by the phrases.

Return STRICT JSON:
{"name": "2-5 word topic name", "description": "one short sentence"}"""


NAME_GROUP_SYSTEM = """Name one mid-level topic group.

Given leaf topic names and representative phrases, return a concise mid-level group name. The name should summarize the shared subject, not copy the largest member.

Return STRICT JSON:
{"name": "2-5 word group name", "description": "one short sentence"}"""


NAME_ASPECT_SYSTEM = """Name one broad top-level aspect.

Given mid-level group names and examples, return a concise top-level aspect name. The name should cover the set as a whole.

Return STRICT JSON:
{"name": "2-5 word aspect name", "description": "one short sentence"}"""


MERGE_TOPIC_SYSTEM = """You find NEAR-DUPLICATE leaf topics that should merge.

You get a numbered list of sibling leaf topics (name + a few example phrases) from one group. Merge ONLY topics that name the SAME narrow thing, just split or worded differently -- e.g. singular vs plural, or two phrasings of one concept. This is de-duplication, NOT categorization.

DO merge: the same specific subject under two names; singular vs plural of one subject; one concept accidentally split in two.
DO NOT merge: two distinct subjects; two sub-kinds or variants of a broader thing; a part vs its whole; two topics that merely share a word. When unsure, do NOT merge.

Return STRICT JSON: {"merge_sets": [[0, 2], [3, 5]]}  (each inner list = ids that are near-duplicates to merge into one). Use only shown integer ids; omit topics that stay alone."""


CONTRAST_SYSTEM = """You refine a set of SIBLING catalogue headings so that none SUBSUMES another.

Each input line is one sibling group under the same parent:
  <id> | <heading> | docs=<count> | members: <a few example member names>

Rename a heading ONLY when EXACTLY ONE other sibling is a STRICTLY NARROWER,
specific kind / subtype / instance of the broad category this heading names, and
this heading's OWN members are the REST of that category. Then rename this broad
heading to state the remainder: "Other <category> (excluding <that one specific
sibling>)". Exclude EXACTLY ONE sibling — the single most specific one it
overlaps. Keep the name short.

DO NOT rename — leave EXACTLY as given — in any of these:
- Two siblings that name roughly the SAME breadth, or are near-synonyms, where
  NEITHER is strictly narrower than the other. They are DUPLICATES to be merged
  elsewhere, not a naming problem. Touch NEITHER of them.
- You would need to exclude more than one sibling to describe the remainder.
- The relationship is merely "related / adjacent", not "X is a specific kind of
  this broad heading".

HARD RULE — renaming is ONE-DIRECTIONAL. If you rename broad heading A to exclude
specific sibling B, then B is the specific one and MUST stay unchanged — do NOT
also rename B. NEVER output a pair where A excludes B and B excludes A, and never
a ring where each heading excludes another. If two headings each look broad
enough to subsume the other, they are synonyms → leave both alone.

If the broad heading you rename has OWN example members that actually belong to
the excluded specific sibling (a misfiled member), list them in "leaks" — those
need regrouping, not renaming.

Return STRICT JSON, integer ids as shown:
{"renames": [{"id": <int broad heading>, "exclude": <int the ONE specific sibling>, "new": "Other X (excluding Y)", "leaks": ["..."]}]}
Include ONLY headings you rename; omit "leaks" when there are none."""

