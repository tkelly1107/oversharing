import time
import json
import streamlit as st

# import your already-defined pieces from the notebook/project
# from detector import Detector, candidates_from_text   # if you want to initialize here
# from your_module import predict_rules, predict_prompt_only, predict_hybrid  # or keep inline wrappers

from predictors import predict_rules, predict_prompt_only, predict_hybrid
st.set_page_config(page_title="Oversharing Risk Demo", page_icon="🔒", layout="centered")
st.markdown("<h2>🔒 Oversharing Privacy Warnings</h2>", unsafe_allow_html=True)
st.write("Type a post below. Choose a backend on the left. Highlights show detected risks; explanations summarize why.")

# Sidebar controls
backend = st.sidebar.selectbox(
    "Backend",
    ["Hybrid (Rules + GPT)", "GPT Prompt-only", "Rules/NER only"],
    index=0
)
pause_s = st.sidebar.slider("Call delay (sec)", 0.0, 0.5, 0.10, 0.05)
st.sidebar.caption("Use a small delay to avoid rate limits when testing repeatedly.")

# Text input
default_text = "Posting from Central Park noon — vibes immaculate."
text = st.text_area("Social post", value=default_text, height=120, max_chars=500)

# Predictors: assume you copied the same functions from your notebook into a small utils module,
# or just copy the three wrappers below to call your in-notebook functions via st.session_state.

if "predictors_ready" not in st.session_state:
    st.session_state.predictors_ready = True

def _predict_dispatch(t: str):
    if backend.startswith("Hybrid"):
        time.sleep(pause_s)
        return predict_hybrid(t)
    if backend.startswith("GPT"):
        time.sleep(pause_s/2)
        return predict_prompt_only(t)
    return predict_rules(t)

LABEL_COLORS = {
    "Location&Time":      "#2E86AB",
    "Contact&IDs":        "#8E44AD",
    "Gov&Financial IDs":  "#D35400",
    "Health&Sensitive":   "#C0392B",
    "Credentials&Security":"#16A085",
    "Workplace/Academic": "#7D6608",
    "Minors":             "#B03A2E",
    "Metadata/Device":    "#566573",
}

def _non_overlapping(spans):
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"]-s["start"])))
    out, last_end = [], -1
    for s in spans:
        if s["start"] >= last_end:
            out.append(s)
            last_end = s["end"]
    return out

def render_block(text: str, pred: dict) -> str:
    import html
    spans = pred.get("spans", [])
    spans = [s for s in spans if 0 <= s["start"] < s["end"] <= len(text)]
    spans = _non_overlapping(spans)

    pieces, cursor = [], 0
    for s in spans:
        if s["start"] > cursor:
            pieces.append(html.escape(text[cursor:s["start"]]))
        frag = html.escape(text[s["start"]:s["end"]])
        label = s.get("label","")
        color = LABEL_COLORS.get(label, "#444")
        pieces.append(
            f'<span style="background:{color}22; border:1px solid {color}; '
            f'padding:0 2px; border-radius:4px;" title="{html.escape(label)}">{frag}</span>'
        )
        cursor = s["end"]
    if cursor < len(text):
        pieces.append(html.escape(text[cursor:]))

    legend_items = "".join(
        f'<div style="display:inline-block;margin:0 8px 6px 0;">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{c}22;'
        f'border:1px solid {c};vertical-align:middle;margin-right:6px;border-radius:3px;"></span>'
        f'<span style="font-size:12px;color:#333;">{html.escape(lbl)}</span></div>'
        for lbl, c in LABEL_COLORS.items()
    )

    expl = pred.get("explanations") or {}
    if not expl and pred.get("labels"):
        for lab in pred["labels"]:
            expl.setdefault(lab, "Potential privacy risk matched by this highlight.")
    expl_html = "".join(
        f'<li><strong style="color:{LABEL_COLORS.get(lab,"#333")}">{html.escape(lab)}:</strong> '
        f'{html.escape(msg)}</li>'
        for lab, msg in expl.items()
    )

    html_block = f"""
    <div style="font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
      <div style="margin-bottom:8px;">{legend_items}</div>
      <div style="font-size:16px; line-height:1.5; padding:10px 12px; background:#fafafa; border:1px solid #eee; border-radius:8px;">
        {''.join(pieces)}
      </div>
      <div style="margin-top:8px;">
        <ul style="margin:8px 0 0 18px; padding:0; font-size:13px; color:#333;">
          {expl_html}
        </ul>
      </div>
    </div>
    """
    return html_block

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Analyze", use_container_width=True):
        st.session_state.last_pred = _predict_dispatch(text)
        st.session_state.last_text = text

with col2:
    st.write(f"**Backend:** {backend}")

if "last_pred" in st.session_state:
    st.markdown(render_block(st.session_state.last_text, st.session_state.last_pred), unsafe_allow_html=True)
else:
    st.caption("Click Analyze to run the selected backend.")