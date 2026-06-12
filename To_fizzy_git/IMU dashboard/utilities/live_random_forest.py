from time import time

from joblib import load
import numpy as np
import pandas as pd
from sklearn import metrics
import time


def _get_final_estimator(model):
    try:
        # scikit-learn Pipeline has `steps`
        if hasattr(model, "steps") and len(model.steps) > 0:
            return model.steps[-1][1]
        # older pipeline-like objects
        if hasattr(model, "named_steps") and len(model.named_steps) > 0:
            return list(model.named_steps.values())[-1]
    except Exception:
        pass
    return model


def _extract_expected_feature_names(model):
    final = _get_final_estimator(model)

    # 1) feature_names_in_ (most common)
    for cand in (final, model):
        if hasattr(cand, "feature_names_in_"):
            try:
                return [str(x) for x in getattr(cand, "feature_names_in_")]
            except Exception:
                pass

    # 2) get_feature_names_out
    for cand in (final, model):
        if hasattr(cand, "get_feature_names_out"):
            try:
                names = cand.get_feature_names_out()
                return [str(x) for x in names]
            except Exception:
                pass

    # 3) common fallback attributes
    for attr in ("feature_list_", "feature_names", "feature_names_", "columns_"):
        if hasattr(model, attr):
            try:
                names = getattr(model, attr)
                return [str(x) for x in list(names)]
            except Exception:
                pass

    return None


def _align_features(single_feature_set: dict, expected_names, fill_value=0.0):
    if expected_names is None:
        return pd.DataFrame([single_feature_set])

    # Create dataframe with expected columns in order, fill missing with fill_value
    df = pd.DataFrame(columns=expected_names, index=[0])
    for k, v in single_feature_set.items():
        if k in df.columns:
            df.at[0, k] = v
    df = df.fillna(fill_value)
    return df


def _get_classes_safe(model, final):
    """Return the first non-None `classes_` attribute from model or final.

    Avoid using boolean `or` on numpy arrays (which raises ValueError).
    """
    for cand in (model, final):
        try:
            cls = getattr(cand, "classes_", None)
        except Exception:
            cls = None
        if cls is not None:
            return cls
    return None


def predict_with_certainty(model, single_feature_set: dict, fill_value=0.0):

    expected = _extract_expected_feature_names(model)
    feature_df = _align_features(single_feature_set, expected, fill_value=fill_value)

    final = _get_final_estimator(model)

    probabilities = None
    classes = None

    # Try model.predict_proba first (works for Pipeline and estimators)
    try:
        proba = model.predict_proba(feature_df)
        probabilities = np.asarray(proba[0], dtype=float)
    except Exception:
        try:
            proba = final.predict_proba(feature_df)
            probabilities = np.asarray(proba[0], dtype=float)
        except Exception:
            # Fall back to predict: give 100% to predicted class
            pred = None
            try:
                pred = model.predict(feature_df)[0]
            except Exception:
                try:
                    pred = final.predict(feature_df)[0]
                except Exception:
                    pred = None

            classes = _get_classes_safe(model, final)
            if classes is None:
                if pred is None:
                    classes = np.array(["unknown"])
                else:
                    classes = np.array([pred])

            probs = np.zeros(len(classes), dtype=float)
            if pred is not None:
                try:
                    idx = list(classes).index(pred)
                    probs[idx] = 1.0
                except ValueError:
                    probs[0] = 1.0
            else:
                probs[0] = 1.0
            probabilities = probs

    if classes is None:
        classes = _get_classes_safe(model, final)
        if classes is None:
            # As a last resort, enumerate positions
            classes = [str(i) for i in range(len(probabilities))]

    result = {str(class_name): round(float(prob) * 100.0, 2) for class_name, prob in zip(classes, probabilities)}

    return result


def predict_and_sort(model, single_feature_set):
    result_dict = predict_with_certainty(model, single_feature_set)
    sorted_results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
