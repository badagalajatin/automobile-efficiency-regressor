import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset(uploaded_file):

    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Only CSV and Excel files are supported.")


def profile_dataset(df):

    profile = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Missing Values": int(df.isnull().sum().sum()),
        "Numeric Columns": len(
            df.select_dtypes(include=np.number).columns
        ),
        "Categorical Columns": len(
            df.select_dtypes(include=["object", "category"]).columns
        )
    }

    return profile


def quality_report(df):

    report = []

    for column in df.columns:

        report.append({
            "Column": column,
            "Data Type": str(df[column].dtype),
            "Missing": int(df[column].isnull().sum()),
            "Unique": int(df[column].nunique()),
            "Missing %": round(
                df[column].isnull().mean() * 100,
                2
            )
        })

    return pd.DataFrame(report)


def detect_outliers(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    output = []

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:

            count = 0

        else:

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = int(
                ((series < lower) | (series > upper)).sum()
            )

        output.append({
            "Column": column,
            "Outliers": count
        })

    return pd.DataFrame(output)


def prepare_features(df, target):

    data = df.copy()

    # Convert object columns that are actually numeric
    for column in data.columns:

        if column == target:
            continue

        if data[column].dtype == "object":

            converted = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.8:
                data[column] = converted

    X = data.drop(columns=[target])
    y = pd.to_numeric(
        data[target],
        errors="coerce"
    )

    # Remove rows with invalid target
    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    return X, y


def train_regression_models(df, target, test_size=0.2):

    X, y = prepare_features(
        df,
        target
    )

    if len(X) < 20:

        raise ValueError(
            "Dataset is too small for reliable regression. "
            "At least 20 valid rows are recommended."
        )

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if len(numeric_features) + len(categorical_features) == 0:

        raise ValueError(
            "No usable predictor columns were found."
        )

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    ridge = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            Ridge(alpha=1.0)
        )
    ])

    lasso = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            Lasso(
                alpha=0.1,
                max_iter=10000
            )
        )
    ])

    ridge.fit(
        X_train,
        y_train
    )

    lasso.fit(
        X_train,
        y_train
    )

    ridge_prediction = ridge.predict(
        X_test
    )

    lasso_prediction = lasso.predict(
        X_test
    )

    ridge_metrics = calculate_metrics(
        y_test,
        ridge_prediction
    )

    lasso_metrics = calculate_metrics(
        y_test,
        lasso_prediction
    )

    # Cross validation
    try:

        ridge_cv = cross_val_score(
            ridge,
            X,
            y,
            cv=5,
            scoring="r2"
        )

        lasso_cv = cross_val_score(
            lasso,
            X,
            y,
            cv=5,
            scoring="r2"
        )

    except Exception:

        ridge_cv = np.array([])
        lasso_cv = np.array([])

    return {
        "ridge_model": ridge,
        "lasso_model": lasso,
        "X_test": X_test,
        "y_test": y_test,
        "ridge_prediction": ridge_prediction,
        "lasso_prediction": lasso_prediction,
        "ridge_metrics": ridge_metrics,
        "lasso_metrics": lasso_metrics,
        "ridge_cv": ridge_cv,
        "lasso_cv": lasso_cv,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "sample_count": len(X)
    }


def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(
                actual,
                predicted
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            )
        ),
        "R2": float(
            r2_score(
                actual,
                predicted
            )
        )
    }


def create_clusters(df, n_clusters=3):

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.shape[1] < 2:

        raise ValueError(
            "At least two numeric columns are required "
            "for vehicle/data profiling."
        )

    # Fill missing values
    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        numeric_df
    )

    # Prevent asking for more clusters than rows
    n_clusters = min(
        n_clusters,
        len(numeric_df)
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        scaled
    )

    result = df.loc[
        numeric_df.index
    ].copy()

    result["Cluster"] = cluster_labels

    summary = (
        result
        .groupby("Cluster")[numeric_df.columns]
        .mean()
        .round(2)
    )

    return result, summary


def generate_plan(df, target):

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    plan = [
        "1. Load and inspect the uploaded dataset.",
        "2. Identify data types, missing values and duplicates.",
        "3. Detect numerical outliers using the IQR method.",
        f"4. Use '{target}' as the prediction target.",
        "5. Automatically identify numeric and categorical predictors.",
        "6. Impute missing values rather than silently dropping them.",
        "7. Standardize numeric features.",
        "8. One-hot encode categorical features.",
        "9. Train Ridge and Lasso regularized regression models.",
        "10. Evaluate models using MAE, RMSE and R².",
        "11. Validate using 5-fold cross-validation.",
        "12. Profile observations using K-Means clustering.",
        "13. Visualize important relationships.",
        "14. Generate evidence-based findings and limitations."
    ]

    return plan, numeric, categorical