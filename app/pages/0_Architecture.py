from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from common import ROOT, annotate, hero, setup

setup("Architecture", "🗺️")
hero("0 · Architecture", "The design this app implements: system HLD, MLOps loop, detection ensemble, evaluation harness and the nightly run. "
     "Drawn before the code was written; every box below has a page in this app.", ["Design → build", "5 diagrams", "Talk track included"])
annotate("How to read this page",
         "The diagrams are hand-drawn SVG from the design document. Dashed lines are control or feedback paths; orange is the alert or rollback path.",
         "The same picture is the production target: one GCP project, BigQuery as the only landing zone, Vertex AI for the model lifecycle, Cloud Run "
         "for serving, Gemini for the narrative layer.")
html = (ROOT / "docs" / "architecture.html").read_text()
components.html(html, height=5200, scrolling=True)
