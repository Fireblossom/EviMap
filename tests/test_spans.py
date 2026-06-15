import unittest

from evimap.spans import build_phrase_index, find_all_occurrences


class SpanTests(unittest.TestCase):
    def test_word_boundary_matching(self):
        text = "SQL appears in SQL logs, but not in NoSQL tooling."
        self.assertEqual(find_all_occurrences(text, "SQL"), [(0, 3), (15, 18)])

    def test_phrase_index_offsets(self):
        docs = [{
            "doc_id": "d1",
            "text": "Bilingual Spanish support and customer support.",
            "metadata": {},
        }]
        extractions = [{
            "doc_id": "d1",
            "phrases": [
                {
                    "text": "Bilingual Spanish",
                    "role_hint": "attribute_or_value",
                    "axis_hints": ["language"],
                },
                {
                    "text": "customer support",
                    "role_hint": "action_or_relation",
                    "axis_hints": ["support"],
                },
            ],
        }]
        entries, occurrences, unmatched = build_phrase_index(docs, extractions)
        self.assertEqual(len(entries), 2)
        self.assertFalse(unmatched)
        self.assertEqual(
            [(o["start"], o["end"]) for o in occurrences],
            [(0, 17), (30, 46)],
        )


if __name__ == "__main__":
    unittest.main()
