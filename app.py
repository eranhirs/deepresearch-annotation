"""Human annotation interface for sentence-level report evaluation.

Annotations are stored in Google Sheets.

Run with:
    streamlit run app.py
"""

import html
import json
import re

import streamlit as st

from components.annotation_instructions import render_annotation_instructions
from utils.annotation_data_loader import (
    get_available_annotation_models,
    get_available_annotation_examples,
    load_annotation_example,
    get_example_files,
    load_example_annotation,
)
from utils.annotation_storage import (
    load_annotations,
    save_annotation,
    download_annotations,
)

st.set_page_config(page_title="Annotate", layout="wide")

# ── Session state defaults ──────────────────────────────────────────────
for key, default in [
    ("annotator_id", ""),
    ("current_sentence_idx", 0),
    ("annotations", {}),
    ("nav_mode", "citation_needed"),
    ("tutorial_mode", False),
    ("tutorial_annotations", {}),
    ("_prev_model", None),
    ("_prev_example", None),
    ("_prev_annotator", None),
    ("_prev_tutorial", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_filtered_segments(segments: list[dict], nav_mode: str) -> list[dict]:
    """Return segments filtered by navigation mode, sorted by section then position."""
    filtered = [s for s in segments if s.get("type") == "text_sentence"]
    if nav_mode == "citation_needed":
        filtered = [s for s in filtered if s.get("is_citation_needed_llm")]
    filtered.sort(key=lambda s: (s.get("section_idx", 0), s.get("idx_in_section", 0)))
    return filtered


def _select_tutorial_subset(segments: list[dict]) -> list[dict]:
    """Select a curated subset of segments covering each issue type for tutorial.

    Picks up to 2 per category, preferring segments where that issue is the ONLY one.
    Returns subset re-sorted by (section_idx, idx_in_section) to preserve report order.
    """
    categories = {
        "no_issues": [],
        "repetitive": [],
        "non_coherent": [],
        "over_specific": [],
        "missing_details": [],
    }
    issue_fields = {
        "repetitive": "llm_is_repetitive",
        "non_coherent": "llm_is_non_coherent",
        "over_specific": "llm_is_over_specific",
        "missing_details": "llm_is_missing_details",
    }

    for seg in segments:
        active_issues = [
            cat for cat, field in issue_fields.items()
            if seg.get(field) is True
        ]
        if not active_issues:
            categories["no_issues"].append((seg, 0))  # 0 = no other issues
        else:
            for cat in active_issues:
                # Count how many OTHER issues this segment has (fewer = cleaner example)
                other_count = len(active_issues) - 1
                categories[cat].append((seg, other_count))

    # Sort each category: prefer segments with fewer other issues (cleaner examples)
    for cat in categories:
        categories[cat].sort(key=lambda x: x[1])

    selected_idxs: set[str] = set()
    selected: list[dict] = []

    for cat in ["non_coherent", "over_specific", "missing_details", "repetitive", "no_issues"]:
        count = 0
        for seg, _ in categories[cat]:
            if count >= 2:
                break
            if seg["idx"] in selected_idxs:
                continue
            selected.append(seg)
            selected_idxs.add(seg["idx"])
            count += 1

    selected.sort(key=lambda s: (s.get("section_idx", 0), s.get("idx_in_section", 0)))
    return selected


def _parse_markdown_to_html(text: str, orig_newlines: str | None = None, is_first: bool = False) -> tuple[str, str]:
    """Minimal markdown-to-HTML for display. Returns (html_text, newline_prefix)."""
    # Compute newline prefix from original answer text
    newlines = ""
    if orig_newlines and not is_first:
        n = max(orig_newlines.count('\n'), 1)
        newlines = '<br/>' * n

    # Bold then italic (order matters: ** before *)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
    # List items: * or -
    text = re.sub(r'^\*\s+', '&bull; ', text)
    text = re.sub(r'^-\s+', '&bull; ', text)
    # Strip --- separator artifacts (standalone or trailing)
    text = re.sub(r'\n*-{3,}\s*$', '', text)
    text = re.sub(r'^-{3,}$', '', text.strip())
    return text, newlines


def _render_report_sections(
    data: dict,
    filtered_segments: list[dict],
    current_seg_idx: str | None,
    annotations: dict,
    *,
    is_tutorial: bool = False,
) -> list[tuple[int, str]]:
    """Build per-section HTML for the report with highlighted sentences.

    Returns list of (section_idx, html_string) tuples.
    """
    sections = data.get("sections", [])
    all_segments = data.get("segments", [])
    answer_text = data.get("answer", "")

    filtered_set = {s["idx"] for s in filtered_segments}

    # Group all segments by section
    by_section: dict[int, list[dict]] = {}
    for seg in all_segments:
        if seg.get("type") != "text_sentence":
            continue
        sec_idx = seg.get("section_idx", 0)
        by_section.setdefault(sec_idx, []).append(seg)
    for sec_idx in by_section:
        by_section[sec_idx].sort(key=lambda s: s.get("idx_in_section", 0))

    result = []
    for section in sections:
        sec_idx = section.get("idx")
        header = section.get("header", "")
        level = section.get("header_level", 2)

        parts = []
        if header:
            header_text = html.escape(header.lstrip("#").strip())
            tag = f"h{min(level, 4)}"
            parts.append(f"<{tag} style='margin-top:1em;margin-bottom:0.3em;'>{header_text}</{tag}>")

        sec_segments = by_section.get(sec_idx, [])
        if not sec_segments and not header:
            continue

        section_start = section.get("start", 0)
        sentence_parts = []
        rendered_count = 0
        for i_seg, seg in enumerate(sec_segments):
            seg_idx = seg["idx"]
            raw_text = seg.get("text", "")

            # Skip header-like segments
            if raw_text.strip().startswith('#'):
                continue

            # Skip degenerate segments (sentencizer artifacts)
            stripped = raw_text.strip()
            if stripped in ('.', '') or re.match(r'^-{2,}$', stripped):
                continue

            escaped = html.escape(raw_text)

            # Compute orig_newlines from answer text
            start_in_sec = seg.get("start_in_section")
            orig_newlines = None
            if start_in_sec is not None and answer_text:
                abs_start = section_start + start_in_sec
                orig_newlines = answer_text[max(0, abs_start - 4):abs_start]

            is_first = (rendered_count == 0)
            text, newlines = _parse_markdown_to_html(escaped, orig_newlines=orig_newlines, is_first=is_first)

            # Skip if text is empty after markdown processing (e.g. stripped --- separators)
            if not text.strip():
                continue

            rendered_count += 1

            in_filter = seg_idx in filtered_set
            is_current = seg_idx == current_seg_idx
            is_annotated = seg_idx in annotations

            if is_current:
                style = "background-color:rgba(253,216,53,0.15); padding:2px 4px; border-radius:4px; font-weight:bold;"
            elif not in_filter and not is_tutorial:
                style = "color:#999;"
            else:
                style = ""

            is_list_item = raw_text.lstrip().startswith(('* ', '- ', '• '))
            if is_list_item:
                # Detect indentation from answer text to preserve nested list levels
                indent_em = 1.5
                if start_in_sec is not None and answer_text:
                    abs_start = section_start + start_in_sec
                    line_start = answer_text.rfind('\n', 0, abs_start)
                    if line_start >= 0:
                        between = answer_text[line_start + 1:abs_start]
                        n_spaces = len(between) - len(between.lstrip(' '))
                        indent_em = 1.5 + n_spaces * 0.75

                sentence_parts.append(
                    f'<div style="margin-left:{indent_em}em; text-indent:-1em; line-height:1.6; margin-top:0.15em; {style}">{text}</div>'
                )
            else:
                sentence_parts.append(f'{newlines}<span style="{style}">{text}</span>')

        if sentence_parts:
            paragraph = " ".join(sentence_parts)
            parts.append(
                f'<div style="line-height:2.2; margin-bottom:16px; padding:10px; '
                f'border:1px solid rgba(128,128,128,0.3); border-radius:8px;">{paragraph}</div>'
            )

        if parts:
            result.append((sec_idx, "\n".join(parts)))

    return result


# ── Sidebar ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Annotation")

    annotator_id = st.text_input("Annotator ID", value=st.session_state.annotator_id)
    st.session_state.annotator_id = annotator_id

    # Tutorial mode toggle
    tutorial_mode = st.checkbox("Tutorial mode", value=st.session_state.tutorial_mode)
    tutorial_changed = tutorial_mode != st.session_state._prev_tutorial
    if tutorial_changed:
        st.session_state.tutorial_mode = tutorial_mode
        st.session_state._prev_tutorial = tutorial_mode
        st.session_state.current_sentence_idx = 0
        st.session_state.tutorial_annotations = {}

    if tutorial_mode:
        # Tutorial: load example files
        example_files = get_example_files()
        if not example_files:
            st.warning("No tutorial examples found. Run export_annotation_data.py with --include-llm-example first.")
            st.stop()

        tutorial_example = st.selectbox("Tutorial example", example_files)

        # Detect changes for tutorial
        example_changed = tutorial_example != st.session_state._prev_example
        if example_changed:
            st.session_state.current_sentence_idx = 0
            st.session_state.tutorial_annotations = {}
            st.session_state._prev_example = tutorial_example

        # Dummy values for non-tutorial code paths
        model = None
        example_id = None
    else:
        models = get_available_annotation_models()
        if not models:
            st.error("No annotation data found in annotation_data/. Run export_annotation_data.py first.")
            st.stop()

        model = st.selectbox("Model", models)
        examples = get_available_annotation_examples(model)
        if not examples:
            st.warning(f"No examples for model {model}")
            st.stop()

        example_id = st.selectbox("Example", examples)

    st.divider()

    nav_default = 1 if st.session_state.nav_mode == "citation_needed" else 0
    nav_mode = st.radio("Navigation mode", ["All sentences", "Citation-needed only"], index=nav_default, horizontal=True)
    nav_mode_key = "all" if nav_mode == "All sentences" else "citation_needed"

    # Detect what changed
    nav_changed = nav_mode_key != st.session_state.nav_mode
    if not tutorial_mode:
        model_changed = model != st.session_state._prev_model
        example_changed = example_id != st.session_state._prev_example
        annotator_changed = annotator_id != st.session_state._prev_annotator
    else:
        model_changed = False
        annotator_changed = annotator_id != st.session_state._prev_annotator

    # Reset sentence index when nav mode, model, or example changes
    if nav_changed or model_changed or example_changed:
        st.session_state.current_sentence_idx = 0
        st.session_state.nav_mode = nav_mode_key
        st.session_state._prev_model = model
        st.session_state._prev_example = example_id if not tutorial_mode else st.session_state._prev_example

    # Reload annotations from sheet when annotator/model/example changes (not in tutorial)
    if not tutorial_mode:
        if annotator_changed or model_changed or example_changed:
            st.session_state._prev_annotator = annotator_id
            st.session_state._prev_model = model
            st.session_state._prev_example = example_id
            if annotator_id:
                try:
                    st.session_state.annotations = load_annotations(annotator_id, model, example_id)
                except Exception as e:
                    st.warning(f"Could not load annotations from sheet: {e}")
                    st.session_state.annotations = {}
            else:
                st.session_state.annotations = {}
    else:
        st.session_state.annotations = {}

    # Instructions
    render_annotation_instructions()

    # Download (not in tutorial)
    if not tutorial_mode and annotator_id:
        if st.button("Download My Annotations"):
            try:
                rows = download_annotations(annotator_id)
                st.download_button(
                    "Save JSON",
                    data=json.dumps(rows, indent=2),
                    file_name=f"annotations_{annotator_id}.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"Download failed: {e}")

# ── Load data ───────────────────────────────────────────────────────────

if tutorial_mode:
    data = load_example_annotation(tutorial_example)
    if data is None:
        st.error(f"Could not load tutorial example: {tutorial_example}")
        st.stop()
else:
    data = load_annotation_example(model, example_id)
    if data is None:
        st.error(f"Could not load annotation data for {model}/{example_id}")
        st.stop()

segments = data.get("segments", [])
filtered = _get_filtered_segments(segments, st.session_state.nav_mode)

if tutorial_mode and filtered:
    filtered = _select_tutorial_subset(filtered)

# Determine current segment (None if filter is empty)
if filtered:
    idx = st.session_state.current_sentence_idx
    idx = max(0, min(idx, len(filtered) - 1))
    st.session_state.current_sentence_idx = idx
    current_seg = filtered[idx]
else:
    current_seg = None
    idx = 0

# Progress bar in sidebar
with st.sidebar:
    if tutorial_mode:
        if filtered:
            tut_completed = sum(1 for s in filtered if s["idx"] in st.session_state.tutorial_annotations)
            tut_total = len(filtered)
            st.progress(tut_completed / tut_total if tut_total > 0 else 0.0)
            st.caption(f"Tutorial: {tut_completed} of {tut_total} completed")
        else:
            st.caption("No tutorial sentences found")
    elif filtered:
        annotated_count = sum(1 for s in filtered if s["idx"] in st.session_state.annotations)
        total_count = len(filtered)
        st.progress(annotated_count / total_count if total_count > 0 else 0.0)
        st.caption(f"Progress: {annotated_count} / {total_count} sentences annotated")

# ── Main area ───────────────────────────────────────────────────────────

# Tutorial banner
if tutorial_mode:
    st.info("**Tutorial** -- Practice annotating each sentence, then submit to see LLM feedback. No annotations are saved.")

# Question
question_text = html.escape(data.get("question", ""))
st.markdown(
    f'<div style="padding:0.75rem 1rem; background-color:rgba(28,131,225,0.1); border-left:4px solid #1c83e1; '
    f'border-radius:4px; margin-bottom:1rem; font-size:1.05em;">'
    f'<b>Question:</b> {question_text}</div>',
    unsafe_allow_html=True,
)

# Warning when filter yields nothing
if not filtered:
    st.warning("No sentences match the current navigation mode. The full report is shown below.")

# ── Navigation (only when we have filtered segments) ────────────────────
if current_seg is not None:
    nav_cols = st.columns([1, 2, 1])
    with nav_cols[0]:
        if st.button("Prev", disabled=idx == 0, use_container_width=True):
            st.session_state.current_sentence_idx = idx - 1
            st.rerun()
    with nav_cols[1]:
        st.markdown(f"<div style='text-align:center;padding-top:8px;'>Sentence {idx + 1} of {len(filtered)}</div>", unsafe_allow_html=True)
    with nav_cols[2]:
        if st.button("Next", disabled=idx == len(filtered) - 1, use_container_width=True):
            st.session_state.current_sentence_idx = idx + 1
            st.rerun()

# ── Rubric definitions (shared between annotation and tutorial) ─────────
rubrics = [
    ("is_repetitive", "repetitive_explanation", "Repetitive?", "llm_is_repetitive", "llm_repetitive_analysis"),
    ("is_non_coherent", "non_coherent_explanation", "Non-coherent?", "llm_is_non_coherent", "llm_coherence_analysis"),
    ("is_over_specific", "over_specific_explanation", "Over-specific?", "llm_is_over_specific", "llm_specificity_analysis"),
    ("is_missing_details", "missing_details_explanation", "Missing details?", "llm_is_missing_details", "llm_missing_details_analysis"),
]

# ── Render report sections with inline annotation ───────────────────────
current_seg_idx = current_seg["idx"] if current_seg is not None else None
current_section_idx = current_seg.get("section_idx") if current_seg is not None else None

annotations_for_display = st.session_state.tutorial_annotations if tutorial_mode else st.session_state.annotations
section_htmls = _render_report_sections(data, filtered, current_seg_idx, annotations_for_display, is_tutorial=tutorial_mode)

for sec_idx, sec_html in section_htmls:
    st.markdown(sec_html, unsafe_allow_html=True)

    # Insert inline annotation panel after the section containing the current sentence
    if current_seg is not None and sec_idx == current_section_idx:
        with st.container(border=True):
            # Compact sentence quote
            sentence_text = current_seg.get("text", "")
            sentence_html, _ = _parse_markdown_to_html(html.escape(sentence_text))
            st.markdown(
                f'<div style="padding:0.5rem 0.75rem; border-left:4px solid #fdd835; '
                f'background-color:rgba(253,216,53,0.08); border-radius:4px; margin-bottom:0.5rem; font-size:0.95em;">'
                f'{sentence_html}</div>',
                unsafe_allow_html=True,
            )

            # Citations
            citations = current_seg.get("citations", [])
            if citations:
                with st.expander(f"Citations ({len(citations)})"):
                    for title, url in citations:
                        st.markdown(f"- [{html.escape(title)}]({url})")

            # Tutorial mode: practice annotation with feedback
            if tutorial_mode:
                seg_idx = current_seg["idx"]
                submitted = st.session_state.tutorial_annotations.get(seg_idx)

                if submitted is None:
                    # ── State A: annotator practices ──
                    st.caption("Evaluate each rubric independently, then submit to see LLM feedback.")
                    tut_annotation = {}
                    for field, _expl_field, label, _llm_field, _llm_analysis_field in rubrics:
                        choice = st.radio(
                            label,
                            ["Yes", "No", "---"],
                            index=2,
                            horizontal=True,
                            key=f"tut_{seg_idx}_{field}",
                        )
                        if choice == "Yes":
                            tut_annotation[field] = True
                        elif choice == "No":
                            tut_annotation[field] = False

                    if st.button("Submit", key="tut_submit", type="primary", use_container_width=True):
                        all_answered = all(f in tut_annotation for f, _, _, _, _ in rubrics)
                        if not all_answered:
                            st.error("Please answer all rubrics before submitting.")
                        else:
                            st.session_state.tutorial_annotations[seg_idx] = tut_annotation
                            st.rerun()

                else:
                    # ── State B: show feedback ──
                    for field, _expl_field, label, llm_field, llm_analysis_field in rubrics:
                        llm_val = current_seg.get(llm_field)
                        llm_analysis = current_seg.get(llm_analysis_field, "")
                        user_val = submitted.get(field)  # True, False, or missing (---)

                        # Format user answer
                        if user_val is True:
                            user_str = "**Yes**"
                        elif user_val is False:
                            user_str = "**No**"
                        else:
                            user_str = "**---**"

                        if llm_val is True:
                            # LLM found an issue — show detail
                            llm_str = ":red[Yes]"
                            st.markdown(f"**{label}** — You: {user_str} · LLM: {llm_str}")
                            if user_val is not True:
                                st.caption("↑ Your answer differs from the LLM on this rubric.")
                            if llm_analysis:
                                st.info(f"**LLM analysis:** {llm_analysis}")
                        else:
                            # LLM found no issue — compact line
                            st.markdown(f"**{label}** — You: {user_str} · LLM: :green[No]")
                            if user_val is True:
                                st.caption("↑ Your answer differs from the LLM on this rubric.")
                                if llm_analysis:
                                    st.info(f"**LLM analysis:** {llm_analysis}")

                    if st.button("Next →", key="tut_next", type="primary", use_container_width=True, disabled=idx == len(filtered) - 1):
                        st.session_state.current_sentence_idx = idx + 1
                        st.rerun()

            # Annotation mode
            else:
                if not annotator_id:
                    st.warning("Enter your Annotator ID in the sidebar to start annotating.")
                else:
                    st.caption("A sentence can have multiple issues. Evaluate each rubric independently.")

                    existing = st.session_state.annotations.get(current_seg["idx"], {})

                    annotation = {}
                    for field, expl_field, label, _llm_field, _llm_analysis_field in rubrics:
                        cols = st.columns([2, 3])
                        existing_val = existing.get(field, "")
                        if existing_val == "True" or existing_val is True:
                            default_idx = 0
                        elif existing_val == "False" or existing_val is False:
                            default_idx = 1
                        else:
                            default_idx = 2

                        with cols[0]:
                            choice = st.radio(
                                label,
                                ["Yes", "No", "---"],
                                index=default_idx,
                                horizontal=True,
                                key=f"rubric_{current_seg['idx']}_{field}",
                            )
                            if choice == "Yes":
                                annotation[field] = True
                            elif choice == "No":
                                annotation[field] = False

                        with cols[1]:
                            annotation[expl_field] = st.text_input(
                                "Explanation (optional)",
                                value=existing.get(expl_field, ""),
                                key=f"expl_{current_seg['idx']}_{field}",
                                label_visibility="collapsed",
                                placeholder="Explanation (optional)",
                            )

                    annotation["other_issues"] = st.text_input(
                        "Other issues",
                        value=existing.get("other_issues", ""),
                        key=f"other_{current_seg['idx']}",
                        placeholder="Any other issues (factual errors, grammar, formatting, etc.)",
                    )

                    # Save & Next
                    if st.button("Save & Next", type="primary", use_container_width=True):
                        has_answer = any(
                            f in annotation for f, _, _, _, _ in rubrics
                        )
                        if not has_answer:
                            st.error("Please answer at least one rubric before saving.")
                        else:
                            try:
                                save_annotation(
                                    annotator_id=annotator_id,
                                    model_name=model,
                                    example_id=example_id,
                                    segment_idx=current_seg["idx"],
                                    sentence_text=current_seg.get("text", ""),
                                    annotation=annotation,
                                )
                                st.session_state.annotations[current_seg["idx"]] = annotation
                                if idx < len(filtered) - 1:
                                    st.session_state.current_sentence_idx = idx + 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to save annotation: {e}")
