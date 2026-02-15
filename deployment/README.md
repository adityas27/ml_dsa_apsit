# Churn Prediction FastAPI deployment

Place your trained pickle file at `deployment/model.pkl`. The pickle should contain either:

- a `dict` with keys: `model` (the trained estimator), `scaler` (fitted StandardScaler or None), and `columns` (list of training columns after one-hot encoding), OR
- a tuple/list in the order `(model, scaler, columns)`, OR
- a single estimator object (pipeline) — in which case scaler/columns may be omitted.

Files:

- `deployment/app.py`: FastAPI application
- `deployment/requirements.txt`: minimal dependencies
- `deployment/example_request.json`: example request body

Run (development):

```bash
pip install -r deployment/requirements.txt
uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
```

Production notes:

- Use a process manager (systemd, docker, or a container orchestrator).
- Run `uvicorn` with multiple workers for CPU-bound tasks: `--workers 4`.
- Ensure `deployment/model.pkl` is present and has the expected structure.

Example request body is in `deployment/example_request.json`.
