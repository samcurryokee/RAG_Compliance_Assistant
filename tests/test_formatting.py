"""Offline tests for the output-format helpers (no database, no API keys).

Run from the project folder:  python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DATABASE_URL", "postgresql://x:y@localhost/z")

import chain  # noqa: E402

STRICT = """Here are 2 questions.

Q1. Which structure forms the roof of the fourth ventricle?
A) Superior medullary velum
B) Tegmentum
C) Corpus callosum
D) Optic chiasm
Answer: A
Explanation: The roof is formed by the superior and inferior medullary vela [Neuroanatomy Lecture Notes Final.pdf, p.24].

Q2. When does the morula enter the uterus?
A) Day 1
B) Day 4
C) Day 9
D) Day 14
Answer: B
Explanation: The morula reaches the uterus about day 4 [Basic Embryology, Pulei.pdf, p.130].
"""

BOLD = """**Q1.** What causes Hirschsprung's disease?
**A)** Failure of neural crest migration
**B)** Failure of notochord formation
**C)** Excess amniotic fluid
**D)** Neural tube closure defect
**Answer:** A
**Explanation:** Neural crest cells fail to reach the distal gut [ALL CONGENITAL ANOMALIES AND THEIR BASIS (2).pdf, p.5].
"""

LOOSE = """### Question 1: Define spina bifida occulta
(a) Swelling with CSF
(b) No swelling, covered with hair
(c) Brain herniation
(d) Open neural tube
Correct answer: (B) No swelling, covered with hair
Rationale: It is covered with skin and may have a tuft of hair
and no cystic swelling [ALL CONGENITAL ANOMALIES AND THEIR BASIS (2).pdf, p.3].
"""


class QuizParsing(unittest.TestCase):
    def test_strict_template(self):
        q = chain.parse_quiz(STRICT)
        self.assertEqual(len(q), 2)
        self.assertEqual([x["answer"] for x in q], ["A", "B"])
        self.assertEqual(len(q[0]["options"]), 4)
        self.assertIn("p.24", q[0]["explanation"])
        self.assertEqual([x["number"] for x in q], [1, 2])

    def test_markdown_bold(self):
        q = chain.parse_quiz(BOLD)
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["answer"], "A")
        self.assertTrue(q[0]["question"].startswith("What causes"))

    def test_loose_variants_and_multiline_explanation(self):
        q = chain.parse_quiz(LOOSE)
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["answer"], "B")
        self.assertIn("no cystic swelling", q[0]["explanation"])
        self.assertEqual(q[0]["options"][1]["letter"], "B")

    def test_count_trims(self):
        self.assertEqual(len(chain.parse_quiz(STRICT, count=1)), 1)

    def test_bad_questions_are_dropped(self):
        no_answer = "Q1. Something?\nA) one\nB) two\nExplanation: x\n"
        wrong_letter = "Q1. Something?\nA) one\nB) two\nAnswer: D\n"
        one_option = "Q1. Something?\nA) one\nAnswer: A\n"
        for text in (no_answer, wrong_letter, one_option, "just prose, no questions"):
            self.assertEqual(chain.parse_quiz(text), [], text)

    def test_one_bad_question_does_not_lose_the_good_one(self):
        mixed = STRICT + "\nQ3. Broken question\nA) only one option\nAnswer: A\n"
        self.assertEqual(len(chain.parse_quiz(mixed)), 2)

    def test_answer_word_is_not_mistaken_for_a_letter(self):
        text = "Q1. Pick one?\nA) x\nB) y\nAnswer: Both are wrong\n"
        self.assertEqual(chain.parse_quiz(text), [])


class TableAndStyle(unittest.TestCase):
    def test_has_table(self):
        good = "Intro\n| Feature | A | Source |\n|---|---|---|\n| x | y | [f.pdf, p.1] |\n"
        aligned = "| A | B |\n| :--- | ---: |\n| 1 | 2 |"
        self.assertTrue(chain.has_table(good))
        self.assertTrue(chain.has_table(aligned))
        self.assertFalse(chain.has_table("No table here | just a pipe"))
        self.assertFalse(chain.has_table("- a\n- b"))

    def test_detect_style(self):
        d = chain.detect_style
        self.assertEqual(d("What is a morula?"), "explain")
        self.assertEqual(d("What is the difference between X and Y?"), "explain")
        self.assertEqual(d("Compare the cranial nerves"), "table")
        self.assertEqual(d("Put that in a table"), "table")
        self.assertEqual(d("Give me 5 MCQs on the basal ganglia"), "mcq")
        self.assertEqual(d("quiz me on embryology"), "mcq")

    def test_detect_count(self):
        self.assertEqual(chain.detect_count("Give me 7 MCQs"), 7)
        self.assertEqual(chain.detect_count("quiz me", default=4), 4)
        self.assertEqual(chain.detect_count("50 questions"), 10)  # capped


class History(unittest.TestCase):
    H = [{"question": "What is a morula?", "answer": "A solid ball of cells."}]

    def test_followup_borrows_previous_question(self):
        self.assertIn("morula", chain.retrieval_query("Now put that in a table", self.H))
        self.assertEqual(chain.retrieval_query("What is the primitive streak?", self.H),
                         "What is the primitive streak?")
        self.assertEqual(chain.retrieval_query("Put that in a table", []), "Put that in a table")

    def test_history_block(self):
        self.assertEqual(chain.history_block([]), "")
        block = chain.history_block(self.H * 3)
        self.assertEqual(block.count("User:"), 2)  # only the last two pairs


if __name__ == "__main__":
    unittest.main()
