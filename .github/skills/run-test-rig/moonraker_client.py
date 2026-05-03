from __future__ import annotations

import json
import http.client
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class MoonrakerClient:
    def __init__(self, base_url: str = "http://klippy-test:7125", timeout_seconds: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}

    def _request_json_with_retry(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        delay = backoff_seconds
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                return self._request_json(method, path, payload, timeout_seconds=timeout_seconds)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.RemoteDisconnected, ConnectionError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2.0, 4.0)

        raise RuntimeError(f"Moonraker request failed {method} {path} {payload}: {last_error}")

    def _request_raw(self, method: str, path: str, timeout_seconds: float | None = None) -> str:
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            method=method,
        )
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")

    def _request_raw_with_retry(
        self,
        method: str,
        path: str,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        timeout_seconds: float | None = None,
    ) -> str:
        delay = backoff_seconds
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                return self._request_raw(method, path, timeout_seconds=timeout_seconds)
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2.0, 4.0)

        raise RuntimeError(f"Moonraker raw request failed {method} {path}: {last_error}")

    def get_printer_info(self) -> dict[str, Any]:
        return self._request_json_with_retry("GET", "/printer/info")

    def get_klippy_log(self) -> str:
        raw = self._request_raw_with_retry("GET", "/server/files/logs/klippy.log")
        idx = raw.rfind("Start printer at ")
        return raw[idx:] if idx >= 0 else raw

    def firmware_restart(self) -> dict[str, Any]:
        return self._request_json_with_retry("POST", "/printer/firmware_restart")

    def run_gcode_script(self, script: str, timeout_seconds: float = 30.0) -> dict[str, Any]:
        # G-code execution is non-idempotent; do not retry/fallback to avoid duplicate command execution.
        return self._request_json_with_retry(
            "POST",
            "/printer/gcode/script",
            {"script": script},
            retries=0,
            timeout_seconds=timeout_seconds,
        )

    def query_objects(self, objects: dict[str, list[str] | None]) -> dict[str, Any]:
        return self._request_json_with_retry("POST", "/printer/objects/query", {"objects": objects})

    def get_gcode_store(self, count: int = 100) -> dict[str, Any]:
        query = urllib.parse.urlencode({"count": count})
        return self._request_json_with_retry("GET", f"/server/gcode_store?{query}")

    def get_state(self) -> str | None:
        info = self.get_printer_info()
        result = info.get("result", {})
        state = result.get("state")
        if isinstance(state, str):
            return state
        return None

    def wait_ready(self) -> bool:
        timeout_seconds = 30.0
        poll_seconds = 0.5
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                state = self.get_state()
            except RuntimeError:
                state = None
            if state == "ready":
                return True
            elif state in ("error", "shutdown"):
                return False
            time.sleep(poll_seconds)
        return False

    def ensure_ready(self) -> None:
        if self.wait_ready():
            return

        try:
            self.firmware_restart()
        except RuntimeError:
            pass

        if not self.wait_ready():
            raise RuntimeError("Moonraker did not reach ready state in time")
