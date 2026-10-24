from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_pipeline(model_name: str) -> Pipeline:
    """Build a consistent TF-IDF text-classification pipeline."""

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=42,
        )
    elif model_name == "linear_svm":
        classifier = LinearSVC(
            max_iter=5000,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("tfidf", tfidf),
        ("classifier", classifier),
    ])