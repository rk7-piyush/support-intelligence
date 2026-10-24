# Support Intelligence and Smart Ticket Routing System

An ML-powered customer-support platform that classifies support messages and
complaint letters, routes them to the right department, and sends uncertain
predictions to human review.

> **Prototype.** Built on the public BANKING77 benchmark, not live bank data.

## Features

- **Ticket Triage**: classify a single message or a full complaint letter,
  view the predicted intent, confidence and routing decision, and work a
  manual-review queue.
- **Batch triage**: classify many tickets at once from a CSV or pasted text,
  with downloadable results.
- **Letter mode**: long letters are cleaned, split into sentences, classified,
  and combined into one ticket with a primary department, secondary
  departments and a priority flag.
- **Department Workload**: see which departments handle auto-routed tickets,
  which received tickets after review, and what is still waiting.
- **Support Analytics**: intent and department distribution, intent explorer
  and keyword search over the dataset.
- **Model Evaluation**: accuracy and F1, per-intent metrics, an interactive
  confidence-threshold explorer, confusion analysis and an error explorer.

## How it works

1. **Classify**: a TF-IDF + calibrated Linear SVM model predicts one of 77
   support intents.
2. **Decide**: the calibrated confidence is compared with a threshold. At or
   above it the ticket is auto-routed, otherwise it goes to manual review.
3. **Route**: intents map to departments through ordered rules in
   `config/intent_routing.json`. Business rules in `src/letter_triage.py`
   cover topics the model was not trained on (loans, fraud, complaints).
4. **Evaluate**: saved test-set results show overall and per-intent quality.

## Results (BANKING77 test set)

| Metric | Value |
|---|---|
| Test accuracy | 88.96% |
| Macro-F1 | 0.890 |
| Intents | 77 |
| Confidence threshold | 55% |
| Auto-routed share (validation) | 81.9% |
| Accuracy of auto-routed tickets (validation) | 95.5% |

## Tech stack

Python, Streamlit, scikit-learn, pandas, Altair

## Project structure

```
support-intelligence/
├── app.py                    # Navigation, theme and sidebar
├── views/                    # Streamlit pages
│   ├── overview.py
│   ├── ticket_triage.py
│   ├── department_workload.py
│   ├── support_analytics.py
│   └── model_evaluation.py
├── ui/                       # Shared UI code
│   ├── theme.py              # CSS and chart helpers
│   ├── data.py               # Cached data loaders
│   └── ledger.py             # Session ticket register
├── src/                      # ML and routing logic
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── routing.py
│   ├── letter_triage.py
│   └── predict.py
├── config/intent_routing.json
├── data/                     # BANKING77 train/test CSVs
├── artifacts/                # Trained model and evaluation outputs
├── tests/test_routing.py
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model (optional)

Run from the project root, in this order:

```bash
python src/data_loader.py   # download and validate BANKING77
python src/train.py         # compare Logistic Regression and Linear SVM
python src/calibrate.py     # calibrate, choose threshold, save final model
python src/evaluate.py      # write test metrics and predictions
```

`calibrate.py` overwrites `artifacts/best_model.joblib`, so always run
`evaluate.py` after it.

## Tests

```bash
pytest
```

## Deployment

Deployed on Streamlit Community Cloud with `app.py` as the main file.
The `data/` and `artifacts/` folders must be committed to the repository.
Keep the `scikit-learn` version in `requirements.txt` the same as the one
used to train the model.

## Limitations

- BANKING77 contains short, single-sentence queries. It is not real support
  data, and it has no loan, fraud or formal-complaint intents.
- Long letters are handled by a rule-based layer on top of the model. Quality
  on real letters has not been measured.
- Model confidence is an estimate, not a guarantee of correctness.
- The review queue and ticket ledger live in the Streamlit session. They are
  not persistent and are not shared between users.
- Evaluation results do not establish production readiness.
