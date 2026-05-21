# Load Testing

Locust harness lives in `load_tests/locustfile.py`.

## Run

```powershell
locust -f load_tests/locustfile.py --host http://localhost:8000
```

Covered routes:

- `/health`
- `/ready`
- `/suppliers`
- `/risk/scores`
- `/alerts`
- `/sentinel/events`

Initial staging targets should be defined with customers, but a reasonable internal pilot target is p95 under 500 ms for read endpoints, under 1% errors, and no tenant leakage under concurrent requests.
