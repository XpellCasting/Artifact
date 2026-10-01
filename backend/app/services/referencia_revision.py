"""Separate completed review, reference origin, and external verification."""


def reference_metadata(annotation):
    status = annotation.get("status", "pending")
    reviewed = status in {"reviewed", "verified"}
    independent = status == "verified"
    return {
        "review_status": "reviewed" if reviewed else "pending",
        "reference_origin": "historical_review",
        "externally_verified": independent,
        "evaluation_scope": "independent_reference" if independent else "historical_review",
        "normality_basis": ("independent_evidence" if independent else "assumed_from_history")
                           if annotation.get("label") == "normal" else "not_applicable",
    }


def describe_annotation(annotation):
    return {**annotation, **reference_metadata(annotation)}
