"""Locust load test suite for staging or local performance checks."""

from __future__ import annotations

from locust import HttpUser, between, task


class SupplierApiUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.headers = {"X-Tenant-ID": "demo-tenant", "X-API-Key": "demo-api-key"}

    @task(5)
    def health(self):
        self.client.get("/health")

    @task(3)
    def ready(self):
        self.client.get("/ready")

    @task(2)
    def suppliers(self):
        self.client.get("/suppliers", headers=self.headers)

    @task(2)
    def risk_scores(self):
        self.client.get("/risk/scores", headers=self.headers)

    @task(2)
    def alerts(self):
        self.client.get("/alerts", headers=self.headers)

    @task(1)
    def sentinel_events(self):
        self.client.get("/sentinel/events", headers=self.headers)
