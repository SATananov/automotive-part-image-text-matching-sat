from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from practical_demo.deployment import DatasetV4DeploymentPredictor


st.set_page_config(
    page_title="Automotive Part Image-Text Checker",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_predictor() -> DatasetV4DeploymentPredictor:
    """Load the deployment model once per Streamlit session."""
    return DatasetV4DeploymentPredictor()


st.title("Automotive Part Image-Text Checker")

st.caption(
    "Optional personal practical experiment ? separate from the official benchmark."
)

st.info(
    "Upload an automotive-part image and enter a short description. "
    "The model compares the visual and textual information and predicts "
    "their relationship."
)

with st.expander("What do the three labels mean?"):
    st.markdown(
        """
- **MATCH** ? image and text describe the same part category.
- **PARTIAL_MATCH** ? different categories from the same functional family.
- **MISMATCH** ? categories belong to different functional families.
"""
    )

uploaded_file = st.file_uploader(
    "1. Upload an automotive-part image",
    type=["jpg", "jpeg", "png", "webp"],
)

image = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width="stretch",
        )

    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")

description = st.text_area(
    "2. Enter a short part description",
    placeholder=(
        "Example: The description names the automotive part "
        "as brake pad."
    ),
    height=100,
)

ready = (
    image is not None
    and bool(description.strip())
)

check = st.button(
    "CHECK RELATION",
    type="primary",
    width="stretch",
    disabled=not ready,
)

if check:
    try:
        with st.spinner(
            "Comparing the image and description..."
        ):
            predictor = load_predictor()

            result = predictor.predict(
                image,
                description,
            )

        st.divider()
        st.subheader("Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted relation",
                result["label"],
            )

        with col2:
            st.metric(
                "Model score",
                f"{100.0 * result['model_score']:.2f}%",
            )

        st.progress(
            float(result["model_score"])
        )

        score_rows = [
            {
                "Relation": label,
                "Model score (%)": round(
                    100.0 * score,
                    2,
                ),
            }
            for label, score in result["scores"].items()
        ]

        score_table = pd.DataFrame(
            score_rows
        ).sort_values(
            "Model score (%)",
            ascending=False,
        )

        st.subheader("Scores for all relations")

        st.dataframe(
            score_table,
            hide_index=True,
            width="stretch",
        )

        st.write(
            "**Recognized text features:**",
            result["recognized_text_features"],
        )

        if result["warning"]:
            st.warning(
                result["warning"]
            )

        st.caption(
            result["note"]
        )

    except Exception as exc:
        st.error(
            "Prediction failed."
        )

        with st.expander(
            "Technical details"
        ):
            st.exception(exc)

st.divider()

st.markdown("### Practical scope")

st.write(
    "This application demonstrates how the research model "
    "could support automotive catalog quality control, for example "
    "by checking whether a product image agrees with its description."
)

st.warning(
    "This is an educational deployment prototype, not a production "
    "vehicle-fitment or safety-critical automotive system."
)

st.caption(
    "The deployment classifier was trained only on Dataset V4 "
    "development data (train + validation) using the previously "
    "frozen model configuration. The locked final test was not used "
    "for deployment training or tuning."
)
