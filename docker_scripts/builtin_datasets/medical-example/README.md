# medical-example

This built-in dataset is a small, synthetic or de-identified medical-record example for the practice walkthrough. It is intended for onboarding and functional testing only.

Container target:

```text
/home/workspace/dataset/medical-example
```

Files:

- `medical_records.json`: raw outpatient-style records as a JSON array. This matches the current data preprocessing script, which scans `*.json` files.

Data policy:

- The sample must not contain real patient identifiers or non-public clinical records.
- Treat it as an API and preprocessing example, not as medical advice or a clinically validated dataset.
- Before publishing a release, remove or replace this sample if its provenance cannot be verified as synthetic or properly de-identified.

Cleaning notes:

- Filled missing diagnosis fields when the history and prescription clearly implied a diagnosis.
- Removed duplicated history content, especially cases where allergy history repeated personal history.
- Normalized empty examination fields to `未提供`.
- Removed malformed leading punctuation in diagnosis text.
- Kept the original Chinese outpatient-record style so preprocessing examples remain realistic.
