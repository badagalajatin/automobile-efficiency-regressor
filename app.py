import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analyzer import (
    load_dataset,
    profile_dataset,
    quality_report,
    detect_outliers,
    generate_plan,
    train_regression_models,
    create_clusters
)


st.set_page_config(
    page_title="Universal Regression & Profiler",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 Universal Regression & Dataset Profiler"
)

st.write(
    "Upload any CSV or Excel dataset. The system automatically "
    "profiles the data, detects quality issues, selects usable "
    "features, trains Ridge and Lasso regression models, "
    "validates predictions and creates statistical clusters."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dataset Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx", "xls"]
)


if uploaded_file is None:

    st.info(
        "👈 Upload a dataset from the sidebar to begin."
    )

    st.stop()


# ============================================================
# LOAD
# ============================================================

try:

    df = load_dataset(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Dataset loading failed: {e}"
    )

    st.stop()


if df.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


# ============================================================
# PROFILE
# ============================================================

st.header("1️⃣ Dataset Profile")

profile = profile_dataset(
    df
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Rows",
    profile["Rows"]
)

c2.metric(
    "Columns",
    profile["Columns"]
)

c3.metric(
    "Missing Values",
    profile["Missing Values"]
)

c4.metric(
    "Duplicates",
    profile["Duplicate Rows"]
)

c5.metric(
    "Numeric Columns",
    profile["Numeric Columns"]
)


st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# QUALITY
# ============================================================

st.header("2️⃣ Data Quality Report")

quality = quality_report(
    df
)

st.dataframe(
    quality,
    use_container_width=True
)


if profile["Missing Values"] > 0:

    st.warning(
        "Missing values detected. The ML pipeline will "
        "impute missing predictor values automatically."
    )

else:

    st.success(
        "No missing values detected."
    )


if profile["Duplicate Rows"] > 0:

    st.warning(
        f"{profile['Duplicate Rows']} duplicate rows detected."
    )

else:

    st.success(
        "No duplicate rows detected."
    )


# ============================================================
# OUTLIERS
# ============================================================

st.header("3️⃣ Numerical Outlier Detection")

outliers = detect_outliers(
    df
)

st.dataframe(
    outliers,
    use_container_width=True
)


# ============================================================
# TARGET SELECTION
# ============================================================

st.header("4️⃣ Select Prediction Target")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


if not numeric_columns:

    st.error(
        "This dataset does not contain a numeric column. "
        "Regression requires a numeric target."
    )

    st.stop()


target = st.selectbox(
    "Choose the numeric column you want to predict:",
    numeric_columns
)


st.success(
    f"Selected target: {target}"
)


# ============================================================
# PLAN
# ============================================================

st.header("5️⃣ Automatic Analysis Plan")

plan, numeric, categorical = generate_plan(
    df,
    target
)

for step in plan:

    st.write(
        "✅",
        step
    )


# ============================================================
# FEATURE DISCOVERY
# ============================================================

st.header("6️⃣ Automatically Detected Features")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Numeric Features")

    for column in numeric:

        if column != target:

            st.write(
                f"• {column}"
            )


with col2:

    st.subheader("Categorical Features")

    if categorical:

        for column in categorical:

            st.write(
                f"• {column}"
            )

    else:

        st.write(
            "No categorical features detected."
        )


# ============================================================
# SETTINGS
# ============================================================

st.header("7️⃣ Model Settings")

col1, col2 = st.columns(2)

with col1:

    test_size = st.slider(
        "Test data percentage",
        min_value=10,
        max_value=40,
        value=20
    )


with col2:

    n_clusters = st.slider(
        "Number of profiling clusters",
        min_value=2,
        max_value=5,
        value=3
    )


# ============================================================
# REGRESSION
# ============================================================

st.header("8️⃣ Regularized Regression")

try:

    results = train_regression_models(
        df,
        target,
        test_size=test_size / 100
    )

except Exception as e:

    st.error(
        f"Regression could not be performed: {e}"
    )

    st.stop()


ridge = results["ridge_metrics"]
lasso = results["lasso_metrics"]


metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R²"
    ],
    "Ridge": [
        ridge["MAE"],
        ridge["RMSE"],
        ridge["R2"]
    ],
    "Lasso": [
        lasso["MAE"],
        lasso["RMSE"],
        lasso["R2"]
    ]
})


st.dataframe(
    metrics.round(4),
    use_container_width=True
)


# ============================================================
# BEST MODEL
# ============================================================

if ridge["R2"] >= lasso["R2"]:

    best_model = "Ridge"
    predictions = results[
        "ridge_prediction"
    ]

else:

    best_model = "Lasso"
    predictions = results[
        "lasso_prediction"
    ]


st.success(
    f"🏆 Best model based on test R²: {best_model}"
)


# ============================================================
# CROSS VALIDATION
# ============================================================

st.header("9️⃣ 5-Fold Cross Validation")

ridge_cv = results["ridge_cv"]
lasso_cv = results["lasso_cv"]


if len(ridge_cv) > 0:

    cv_table = pd.DataFrame({
        "Fold": range(1, 6),
        "Ridge R²": ridge_cv,
        "Lasso R²": lasso_cv
    })

    st.dataframe(
        cv_table.round(4),
        use_container_width=True
    )

    st.write(
        f"Average Ridge CV R²: "
        f"{ridge_cv.mean():.4f}"
    )

    st.write(
        f"Average Lasso CV R²: "
        f"{lasso_cv.mean():.4f}"
    )

else:

    st.warning(
        "Cross-validation could not be calculated."
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.header("🔟 Actual vs Predicted")

actual = results[
    "y_test"
].values


fig, ax = plt.subplots()

ax.scatter(
    actual,
    predictions
)

minimum = min(
    actual.min(),
    predictions.min()
)

maximum = max(
    actual.max(),
    predictions.max()
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

ax.set_xlabel(
    f"Actual {target}"
)

ax.set_ylabel(
    f"Predicted {target}"
)

ax.set_title(
    f"{best_model}: Actual vs Predicted {target}"
)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# CLUSTERING
# ============================================================

st.header("1️⃣1️⃣ Dataset / Vehicle Profiling")

try:

    clustered_df, cluster_summary = create_clusters(
        df,
        n_clusters=n_clusters
    )

    st.subheader(
        "Cluster Characteristics"
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

except Exception as e:

    st.warning(
        f"Clustering unavailable: {e}"
    )

    clustered_df = None


# ============================================================
# CLUSTER VISUALIZATION
# ============================================================

if clustered_df is not None:

    numeric_for_cluster = clustered_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_for_cluster = [
        column
        for column in numeric_for_cluster
        if column != target
    ]

    if len(numeric_for_cluster) >= 2:

        x_feature = st.selectbox(
            "X-axis feature",
            numeric_for_cluster,
            index=0
        )

        y_feature = st.selectbox(
            "Y-axis feature",
            numeric_for_cluster,
            index=1
        )

        fig, ax = plt.subplots()

        for cluster in sorted(
            clustered_df["Cluster"].unique()
        ):

            data = clustered_df[
                clustered_df["Cluster"] == cluster
            ]

            ax.scatter(
                data[x_feature],
                data[y_feature],
                label=f"Cluster {cluster}"
            )

        ax.set_xlabel(
            x_feature
        )

        ax.set_ylabel(
            y_feature
        )

        ax.set_title(
            "Automatically Generated Data Profiles"
        )

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# INSIGHTS
# ============================================================

st.header("1️⃣2️⃣ Automated Findings")

st.write(
    f"**Finding 1:** The dataset contains "
    f"{len(df):,} observations and "
    f"{len(df.columns)} columns."
)

st.write(
    f"**Finding 2:** `{target}` was selected as "
    f"the prediction target."
)

st.write(
    f"**Finding 3:** {best_model} produced the strongest "
    f"test-set R² among the two regularized models."
)

st.write(
    f"**Finding 4:** The regression pipeline used "
    f"{len(numeric) - (1 if target in numeric else 0)} "
    f"numeric predictors and "
    f"{len(categorical)} categorical predictors."
)

if profile["Missing Values"] > 0:

    st.write(
        "**Finding 5:** Missing values were detected and "
        "handled through model-pipeline imputation."
    )

else:

    st.write(
        "**Finding 5:** No missing values were detected."
    )


# ============================================================
# LIMITATIONS
# ============================================================

st.header("1️⃣3️⃣ Limitations & Confidence")

limitations = [
    "Regression requires a numeric target variable.",
    "High R² does not prove causation.",
    "Small datasets can produce unstable model estimates.",
    "K-Means clusters are statistical groups, not necessarily official business or industry categories.",
    "Outliers may represent legitimate observations rather than data errors.",
    "Predictions outside the training distribution may be unreliable.",
    "Model performance depends on the quality and relevance of the uploaded features."
]

for limitation in limitations:

    st.write(
        "•",
        limitation
    )


# ============================================================
# FINAL
# ============================================================

st.success(
    "🎉 Analysis complete: "
    "Upload → Profile → Quality Check → "
    "Target Selection → Feature Detection → "
    "Ridge/Lasso → Cross Validation → "
    "Prediction Validation → Clustering → "
    "Insights → Limitations"
)