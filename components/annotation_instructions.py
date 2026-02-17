"""Annotation rubric instructions for the sidebar.

Adapted from the LLM prompts in src/score_sentence_level_rubrics.py.
"""

import streamlit as st


def render_annotation_instructions() -> None:
    """Render the annotation guidelines in an expandable sidebar section."""
    with st.expander("Annotation Instructions", expanded=False):
        st.markdown("""
**A sentence can have multiple issues.** Evaluate each rubric independently.

---

**Repetitive?**
Does the sentence unnecessarily repeat information already clearly stated earlier?

Mark **Yes** if:
- It repeats information already clearly stated in previous sentences
- It conveys the same information using different wording, without adding new insights

Mark **No** if:
- This is the first mention of the information
- The sentence explicitly references a prior mention (e.g., "as mentioned above")
- It summarizes previous points in a conclusion
- It outlines points to be discussed (introduction / section overview)

---

**Non-coherent?**
Does the sentence NOT logically fit within the surrounding text?

Mark **Yes** if:
- **Domain drift:** introduces terminology or assumptions that conflict with the established domain
- **Context hallucination:** discusses entities or events never introduced
- **Broken narrative:** jumps to a new topic without a connective transition

Mark **No** if the sentence stays on topic, supports logical progression, and flows smoothly.

---

**Over-specific?**
Does the sentence provide detail that exceeds what the question requires?

Mark **Yes** if:
- **Tangential minutiae:** includes technical details not needed to understand the core concept
- **Mismatched scope:** the question is high-level but the sentence dives into niche domains
- **Exhaustive listing:** lists exhaustive examples when a summary would suffice

Mark **No** if the details serve as necessary evidence, resolve ambiguity, or match domain-standard depth. Do NOT flag facts that serve as proof for a claim.

---

**Missing details?**
Does the sentence introduce a concept, claim, or method that the rest of the response fails to adequately elaborate on?

Mark **Yes** if:
- It introduces a concept/method/finding not explained elsewhere in the response
- It references a result without enough context to understand it
- It uses jargon never defined or clarified in the response

Mark **No** if:
- The concept is elaborated elsewhere in the response
- The sentence is self-contained (simple factual statement)
- It is part of a conclusion referencing previously discussed material

---

**Other issues:** Free-text field for anything else (factual errors, grammar, formatting, etc.)
""")
