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
    ("nav_mode", "all"),
    ("tutorial_mode", False),
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


def _parse_markdown_to_html(text: str, orig_newlines: str | None = None, is_first: bool = False) -> tuple[str, str]:
    """Minimal markdown-to-HTML for display. Returns (html_text, newline_prefix)."""
    # Compute newline prefix from original answer text
    newlines = ""
    if orig_newlines and not is_first:
        n = max(orig_newlines.count('\n'), 1)
        newlines = '<br/>' * n

    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    # List items: * or -
    text = re.sub(r'^\*\s+', '&bull; ', text)
    text = re.sub(r'^-\s+', '&bull; ', text)
    # Strip --- separator artifacts
    text = re.sub(r'^-{3,}$', '', text.strip())
    return text, newlines


def _render_report_html(
    data: dict,
    filtered_segments: list[dict],
    current_seg_idx: str,
    annotations: dict,
) -> str:
    """Build the HTML for the report view with numbered, highlighted sentences."""
    sections = data.get("sections", [])
    all_segments = data.get("segments", [])
    answer_text = data.get("answer", "")

    # Build lookup: segment idx -> sequential number in filtered list
    filtered_set = {s["idx"] for s in filtered_segments}
    seg_number = {}
    for i, s in enumerate(filtered_segments):
        seg_number[s["idx"]] = i + 1

    # Group all segments by section
    by_section: dict[int, list[dict]] = {}
    for seg in all_segments:
        if seg.get("type") != "text_sentence":
            continue
        sec_idx = seg.get("section_idx", 0)
        by_section.setdefault(sec_idx, []).append(seg)
    for sec_idx in by_section:
        by_section[sec_idx].sort(key=lambda s: s.get("idx_in_section", 0))

    parts = []
    for section in sections:
        sec_idx = section.get("idx")
        header = section.get("header", "")
        level = section.get("header_level", 2)

        if header:
            header_text = html.escape(header.lstrip("#").strip())
            tag = f"h{min(level, 4)}"
            parts.append(f"<{tag} style='margin-top:1em;margin-bottom:0.3em;'>{header_text}</{tag}>")

        sec_segments = by_section.get(sec_idx, [])
        if not sec_segments:
            continue

        section_start = section.get("start", 0)
        sentence_parts = []
        for i_seg, seg in enumerate(sec_segments):
            seg_idx = seg["idx"]
            raw_text = seg.get("text", "")

            # Skip header-like segments
            if raw_text.strip().startswith('#'):
                continue

            escaped = html.escape(raw_text)

            # Compute orig_newlines from answer text
            start_in_sec = seg.get("start_in_section")
            orig_newlines = None
            if start_in_sec is not None and answer_text:
                abs_start = section_start + start_in_sec
                orig_newlines = answer_text[max(0, abs_start - 4):abs_start]

            is_first = (i_seg == 0)
            text, newlines = _parse_markdown_to_html(escaped, orig_newlines=orig_newlines, is_first=is_first)

            in_filter = seg_idx in filtered_set
            is_current = seg_idx == current_seg_idx
            is_annotated = seg_idx in annotations

            num = seg_number.get(seg_idx)
            if is_annotated and num is not None:
                prefix = f"[&#10003;{num}]"
            elif num is not None:
                prefix = f"[{num}]"
            else:
                prefix = ""

            if is_current:
                style = "border-left:4px solid #fdd835; padding-left:6px; background-color:rgba(253,216,53,0.15); font-weight:bold;"
            elif is_annotated:
                style = "border-left:3px solid #4caf50; padding-left:4px;"
            elif not in_filter:
                style = "color:#999;"
            else:
                style = ""

            sentence_parts.append(f'{newlines}<span style="{style}">{prefix} {text}</span>')

        paragraph = " ".join(sentence_parts)
        parts.append(
            f'<div style="line-height:2.2; margin-bottom:16px; padding:10px; '
            f'border:1px solid rgba(128,128,128,0.3); border-radius:8px;">{paragraph}</div>'
        )

    return "\n".join(parts)


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

    nav_mode = st.radio("Navigation mode", ["All sentences", "Citation-needed only"], horizontal=True)
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

if not filtered:
    st.warning("No sentences match the current navigation mode.")
    st.stop()

# Progress bar in sidebar
with st.sidebar:
    if tutorial_mode:
        st.caption(f"{len(filtered)} sentences in this example")
    else:
        annotated_count = sum(1 for s in filtered if s["idx"] in st.session_state.annotations)
        total_count = len(filtered)
        st.progress(annotated_count / total_count if total_count > 0 else 0.0)
        st.caption(f"Progress: {annotated_count} / {total_count} sentences annotated")

# ── Main area ───────────────────────────────────────────────────────────

# Tutorial banner
if tutorial_mode:
    st.info("**Tutorial** -- Review the example annotations to understand the rubrics. No annotations are saved in this mode.")

# Question
st.markdown(f"**Question:** {data.get('question', '')}")

# Clamp index
idx = st.session_state.current_sentence_idx
idx = max(0, min(idx, len(filtered) - 1))
st.session_state.current_sentence_idx = idx
current_seg = filtered[idx]

# ── Navigation ─────────────────────────────────────────────────────────
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

# ── Current sentence display ───────────────────────────────────────────
sentence_text = current_seg.get("text", "")
sentence_html, _ = _parse_markdown_to_html(html.escape(sentence_text))
st.markdown(
    f'<div style="padding:1rem; border-left:4px solid #fdd835; '
    f'background-color:rgba(253,216,53,0.1); border-radius:0.5rem; margin-bottom:1rem;">'
    f'{sentence_html}</div>',
    unsafe_allow_html=True,
)

# Show citations if present
citations = current_seg.get("citations", [])
if citations:
    with st.expander(f"Citations ({len(citations)})"):
        for title, url in citations:
            st.markdown(f"- [{html.escape(title)}]({url})")

# ── Annotation / Tutorial panel ────────────────────────────────────────

# Rubric definitions (shared between annotation and tutorial)
rubrics = [
    ("is_repetitive", "repetitive_explanation", "Repetitive?", "llm_is_repetitive", "llm_repetitive_analysis"),
    ("is_non_coherent", "non_coherent_explanation", "Non-coherent?", "llm_is_non_coherent", "llm_coherence_analysis"),
    ("is_over_specific", "over_specific_explanation", "Over-specific?", "llm_is_over_specific", "llm_specificity_analysis"),
    ("is_missing_details", "missing_details_explanation", "Missing details?", "llm_is_missing_details", "llm_missing_details_analysis"),
]

if tutorial_mode:
    # Tutorial: show LLM answers read-only
    st.subheader("LLM Annotations (read-only)")
    for field, expl_field, label, llm_field, llm_analysis_field in rubrics:
        llm_val = current_seg.get(llm_field)
        llm_analysis = current_seg.get(llm_analysis_field, "")

        if llm_val is True:
            badge = ":red[**Yes**]"
        elif llm_val is False:
            badge = ":green[**No**]"
        else:
            badge = ":gray[N/A]"

        st.markdown(f"**{label}** {badge}")
        if llm_analysis:
            with st.expander("LLM analysis"):
                st.markdown(llm_analysis)

    # Tutorial "Next" button (no saving)
    if st.button("Next", key="tutorial_next", type="primary", use_container_width=True, disabled=idx == len(filtered) - 1):
        st.session_state.current_sentence_idx = idx + 1
        st.rerun()

else:
    # Regular annotation mode
    if not annotator_id:
        st.warning("Enter your Annotator ID in the sidebar to start annotating.")
        st.stop()

    st.caption("A sentence can have multiple issues. Evaluate each rubric independently.")

    # Load existing annotation for this segment
    existing = st.session_state.annotations.get(current_seg["idx"], {})

    annotation = {}
    for field, expl_field, label, _llm_field, _llm_analysis_field in rubrics:
        cols = st.columns([2, 3])
        existing_val = existing.get(field, "")
        # Map stored string values to radio index
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
            # else leave unset

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
        # Require at least one rubric answered
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

# ── Full report (context) ──────────────────────────────────────────────
st.divider()
with st.expander("Full Report (context)", expanded=True):
    report_html = _render_report_html(data, filtered, current_seg["idx"], st.session_state.annotations)
    st.markdown(report_html, unsafe_allow_html=True)
