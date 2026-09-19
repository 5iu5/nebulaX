"""Shared presentation helpers for the Streamlit experience layer."""

from __future__ import annotations

from html import escape

import streamlit as st


_SEVERITY_CLASS = {
    "Normal": "status-normal",
    "Advisory": "status-advisory",
    "High": "status-high",
    "Critical": "status-critical",
}


def page_header(*, eyebrow: str, title: str, description: str, tags: tuple[str, ...] = ()) -> None:
    """Render a consistent subsystem hero without exposing model internals."""
    chips = "".join(f'<span class="hero-chip">{escape(tag)}</span>' for tag in tags)
    st.markdown(
        f"""
        <section class="page-hero">
          <div class="page-eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(description)}</p>
          <div class="hero-chips">{chips}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def upload_intro(title: str, description: str) -> None:
    """Render the primary upload call-to-action above a file picker."""
    st.markdown(
        f"""
        <div class="upload-intro">
          <div><h2>{escape(title)}</h2><p>{escape(description)}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(severity: str, label: str | None = None) -> None:
    """Render an accessible operational-severity badge."""
    css_class = _SEVERITY_CLASS.get(severity, "status-advisory")
    text = label or severity
    st.markdown(
        f'<span class="status-badge {css_class}"><span aria-hidden="true" class="status-dot"></span>{escape(text)}</span>',
        unsafe_allow_html=True,
    )


def section_intro(title: str, description: str | None = None) -> None:
    """Render a compact section heading with optional supporting copy."""
    body = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(f'<div class="section-intro"><h2>{escape(title)}</h2>{body}</div>', unsafe_allow_html=True)


_DEFAULT_STEP = 2


def set_step(step: int) -> None:
    """Record how far the operator has progressed, for the sidebar stepper."""
    st.session_state["nx_step"] = step


def workflow_steps(labels: tuple[str, ...]) -> None:
    """Render the sidebar workflow with completed/active state.

    Called after the page body so it reflects the current run, not the previous one.
    """
    current = int(st.session_state.get("nx_step", _DEFAULT_STEP))
    items = []
    for index, label in enumerate(labels, start=1):
        css = "is-done" if index < current else "is-active" if index == current else ""
        items.append(f'<div class="{css}">{escape(label)}</div>')
    st.markdown('<div class="workflow">' + "".join(items) + "</div>", unsafe_allow_html=True)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def uploaded_files_list(files, key: str):
    """List the selected files as rows under the upload box, each with a remove button.

    Streamlit draws its own chips inside the dropzone; those are hidden so the list reads
    as a normal list. Streamlit will not let us mutate the uploader's value, so removal is
    tracked separately here and the kept files are returned for the caller to analyse.
    """
    items = list(files) if isinstance(files, (list, tuple)) else ([files] if files else [])
    state_key = f"_removed_{key}"
    removed = st.session_state.setdefault(state_key, set())
    # Drop stale entries so re-uploading a previously removed name works.
    removed &= {f.name for f in items}
    st.session_state[state_key] = removed

    kept = [f for f in items if f.name not in removed]
    if not kept:
        return []

    # Emitted here, not in the global stylesheet: hiding Streamlit's in-dropzone chips is
    # only safe when these rows actually render. Kept together, the two can never drift
    # out of sync (e.g. when a cached page module leaves the rows behind).
    st.markdown(
        "<style>"
        '[data-testid="stFileChip"] {display:none!important;}'
        # Direct-child scoping keeps this off the page-level vertical rhythm.
        '[data-testid="stVerticalBlock"]:has(> [data-testid="stLayoutWrapper"] > '
        '[data-testid="stHorizontalBlock"] .file-row) {gap:.3rem!important;}'
        # The button is its own column, so it is overlaid onto the card to sit inside it.
        '[data-testid="stHorizontalBlock"]:has(.file-row) {position:relative;gap:0;min-height:42px;align-items:center;}'
        '[data-testid="stHorizontalBlock"]:has(.file-row) [data-testid="stElementContainer"] {margin:0;}'
        '[data-testid="stHorizontalBlock"]:has(.file-row) > [data-testid="stColumn"]:first-child {'
        "flex:1 1 100%;width:100%;min-width:0;}"
        '[data-testid="stHorizontalBlock"]:has(.file-row) > [data-testid="stColumn"]:last-child {'
        "position:absolute;right:.35rem;top:8px;width:auto;min-width:0;padding:0;"
        "display:flex;align-items:center;z-index:2;}"
        '[data-testid="stHorizontalBlock"]:has(.file-row) > [data-testid="stColumn"]:last-child button {'
        "height:42px;width:34px;min-height:0;padding:0;margin:0;border:none;background:transparent;"
        "box-shadow:none;border-radius:7px;color:var(--muted);font-size:.85rem;line-height:1;"
        "display:flex;align-items:center;justify-content:center;}"
        '[data-testid="stHorizontalBlock"]:has(.file-row) > [data-testid="stColumn"]:last-child button:hover {'
        "background:#fdeced;color:var(--high);}"
        "</style>",
        unsafe_allow_html=True,
    )
    # Rows live in their own container so their spacing can be tightened without
    # touching the page-level vertical rhythm, and so a long list can scroll.
    many = len(kept) > 6
    box = st.container(height=308, border=False) if many else st.container()
    with box:
        _render_file_rows(kept, key, removed, state_key)
    return kept


def _render_file_rows(kept, key, removed, state_key) -> None:
    for f in kept:
        name_col, remove_col = st.columns([12, 1], vertical_alignment="center")
        name_col.markdown(
            f'<div class="file-row"><span class="fr-icon" aria-hidden="true">\u2913</span>'
            f'<span class="fr-name">{escape(f.name)}</span>'
            f'<span class="fr-size">{_human_size(getattr(f, "size", 0))}</span></div>',
            unsafe_allow_html=True,
        )
        if remove_col.button("✕", key=f"rm_{key}_{f.name}", help=f"Remove {f.name}"):
            removed.add(f.name)
            st.session_state[state_key] = removed
            st.rerun()
