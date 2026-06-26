"""
The abstraction layer for printer communication.

MOCK mode  : returns safe status codes so the full pipeline can be tested
             without a physical printer (default).
LIVE mode  : sends real HTTP/API requests to the printer firmware to check
             status and pause the print on a confirmed defect.

The deployment target (Elegoo OrangeStorm Giga) runs Klipper, which is driven
by the Moonraker REST API, so MOONRAKER is the default live firmware. OctoPrint
is also supported for Marlin-based machines.

Configuration is read from environment variables so that main.py can keep
constructing `PrinterInterface()` with no arguments:

    PRINTER_MODE     mock | live                      (default: mock)
    PRINTER_FIRMWARE moonraker | octoprint            (default: moonraker)
    PRINTER_URL      e.g. http://192.168.1.50:7125    (Moonraker default port 7125)
                          http://octopi.local         (OctoPrint)
    PRINTER_API_KEY  Moonraker: only if [authorization] requires it
                     OctoPrint: always required

Example (Giga / Moonraker):
    PRINTER_MODE=live PRINTER_URL=http://192.168.1.50:7125 python main.py --source 0
"""

import os

PRINTER_MODE = os.environ.get("PRINTER_MODE", "mock").lower()
PRINTER_FIRMWARE = os.environ.get("PRINTER_FIRMWARE", "moonraker").lower()

# Klipper/Moonraker states that mean a job is actively running.
_MOONRAKER_PRINTING_STATES = {"printing"}
# OctoPrint reports a human-readable state string.
_OCTOPRINT_PRINTING_STATE = "Printing"

# Timeout (seconds) for every HTTP call. Short, so a flaky network never
# stalls the monitoring loop.
_HTTP_TIMEOUT = 3


class PrinterInterface:
    """
    Unified interface for communicating with the printer firmware.

    Mock mode: logs actions to stdout, safe to run anywhere.
    Live mode (Moonraker / OctoPrint): sends HTTP API requests.
    """

    def __init__(self, api_url: str = "", api_key: str = "", firmware: str = ""):
        """
        Args:
            api_url:  Base URL of Moonraker or OctoPrint. Falls back to the
                      PRINTER_URL environment variable when empty.
            api_key:  API key. Falls back to PRINTER_API_KEY. Required for
                      OctoPrint; for Moonraker only when its [authorization]
                      section is configured to require one.
            firmware: 'moonraker' or 'octoprint'. Falls back to
                      PRINTER_FIRMWARE (default 'moonraker').
        """
        self.api_url = (api_url or os.environ.get("PRINTER_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("PRINTER_API_KEY", "")
        self.firmware = (firmware or PRINTER_FIRMWARE).lower()
        self._mode = PRINTER_MODE

        print(f"[PrinterInterface] Mode: {self._mode.upper()}")
        if self._mode == "live":
            print(f"[PrinterInterface] Firmware: {self.firmware.upper()} @ {self.api_url or '(no URL set!)'}")
            if not self.api_url:
                print("[PrinterInterface] WARNING: PRINTER_URL is empty; live calls will fail.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_printing(self) -> bool:
        """Returns True if the printer is currently printing."""
        if self._mode == "mock":
            print("[PrinterInterface] MOCK: is_printing() -> True")
            return True
        if self.firmware == "octoprint":
            return self._octoprint_is_printing()
        return self._moonraker_is_printing()

    def pause_print(self) -> bool:
        """
        Pause the active print job.

        Returns:
            True if the pause command was accepted, False otherwise.
        """
        if self._mode == "mock":
            print("[PrinterInterface] MOCK: pause_print() -> paused")
            return True
        if self.firmware == "octoprint":
            return self._octoprint_pause()
        return self._moonraker_pause()

    def resume_print(self) -> bool:
        """
        Resume a paused print job.

        Returns:
            True if the resume command was accepted, False otherwise.
        """
        if self._mode == "mock":
            print("[PrinterInterface] MOCK: resume_print() -> resumed")
            return True
        if self.firmware == "octoprint":
            return self._octoprint_resume()
        return self._moonraker_resume()

    def cancel_print(self) -> bool:
        """
        Cancel/stop the active print job. This is irreversible on the printer side.

        Returns:
            True if the cancel command was accepted, False otherwise.
        """
        if self._mode == "mock":
            print("[PrinterInterface] MOCK: cancel_print() -> cancelled")
            return True
        if self.firmware == "octoprint":
            return self._octoprint_cancel()
        return self._moonraker_cancel()

    def ping(self) -> bool:
        """
        Lightweight reachability check, useful for on-site setup: confirms the
        firmware API answers before a real print is started. Returns True on a
        successful response.
        """
        if self._mode == "mock":
            print("[PrinterInterface] MOCK: ping() -> True")
            return True
        try:
            import requests

            if self.firmware == "octoprint":
                url, headers = f"{self.api_url}/api/version", self._octoprint_headers()
            else:
                url, headers = f"{self.api_url}/printer/info", self._moonraker_headers()

            r = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
            ok = r.status_code == 200
            print(f"[PrinterInterface] ping {url} -> {r.status_code}")
            return ok
        except Exception as e:
            print(f"[PrinterInterface] ERROR pinging printer: {e}")
            return False

    # ------------------------------------------------------------------
    # Moonraker (Klipper) implementation  --  default, used by the Giga
    # ------------------------------------------------------------------

    def _moonraker_headers(self) -> dict:
        # Moonraker accepts X-Api-Key only if [authorization] is configured.
        return {"X-Api-Key": self.api_key} if self.api_key else {}

    def _moonraker_is_printing(self) -> bool:
        try:
            import requests

            r = requests.get(
                f"{self.api_url}/printer/objects/query?print_stats",
                headers=self._moonraker_headers(),
                timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
            state = r.json()["result"]["status"]["print_stats"]["state"]
            return state in _MOONRAKER_PRINTING_STATES
        except Exception as e:
            print(f"[PrinterInterface] ERROR querying Moonraker: {e}")
            return False

    def _moonraker_pause(self) -> bool:
        try:
            import requests

            r = requests.post(
                f"{self.api_url}/printer/print/pause",
                headers=self._moonraker_headers(),
                timeout=_HTTP_TIMEOUT,
            )
            ok = r.status_code == 200
            print(f"[PrinterInterface] Moonraker pause -> {r.status_code}")
            return ok
        except Exception as e:
            print(f"[PrinterInterface] ERROR pausing via Moonraker: {e}")
            return False

    def _moonraker_resume(self) -> bool:
        try:
            import requests

            r = requests.post(
                f"{self.api_url}/printer/print/resume",
                headers=self._moonraker_headers(),
                timeout=_HTTP_TIMEOUT,
            )
            ok = r.status_code == 200
            print(f"[PrinterInterface] Moonraker resume -> {r.status_code}")
            return ok
        except Exception as e:
            print(f"[PrinterInterface] ERROR resuming via Moonraker: {e}")
            return False

    def _moonraker_cancel(self) -> bool:
        try:
            import requests

            r = requests.post(
                f"{self.api_url}/printer/print/cancel",
                headers=self._moonraker_headers(),
                timeout=_HTTP_TIMEOUT,
            )
            ok = r.status_code == 200
            print(f"[PrinterInterface] Moonraker cancel -> {r.status_code}")
            return ok
        except Exception as e:
            print(f"[PrinterInterface] ERROR cancelling via Moonraker: {e}")
            return False

    # ------------------------------------------------------------------
    # OctoPrint (Marlin) implementation
    # ------------------------------------------------------------------

    def _octoprint_headers(self) -> dict:
        return {"X-Api-Key": self.api_key}

    def _octoprint_is_printing(self) -> bool:
        try:
            import requests

            r = requests.get(
                f"{self.api_url}/api/job",
                headers=self._octoprint_headers(),
                timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
            return r.json().get("state", "") == _OCTOPRINT_PRINTING_STATE
        except Exception as e:
            print(f"[PrinterInterface] ERROR querying OctoPrint: {e}")
            return False

    def _octoprint_pause(self) -> bool:
        try:
            import requests

            # action 'pause' always pauses (vs 'toggle', which could resume).
            r = requests.post(
                f"{self.api_url}/api/job",
                json={"command": "pause", "action": "pause"},
                headers=self._octoprint_headers(),
                timeout=_HTTP_TIMEOUT,
            )
            ok = r.status_code == 204
            print(f"[PrinterInterface] OctoPrint pause -> {r.status_code}")
            return ok
        except Exception as e:
            print(f"[PrinterInterface] ERROR pausing via OctoPrint: {e}")
            return False

    def _octoprint_resume(self) -> bool:
        try:
            import requests

            r = requests.post(
                f"{self.api_url}/api/job",
                json={"command": "pause", "action": "resume"},
                headers=self._octoprint_headers(),
                timeout=_HTTP_TIMEOUT,
            )
            ok = r.status_code == 204
            print(f"[PrinterInterface] OctoPrint resume -> {r.status_code}")
            return ok
        except Exception as e:
            print(f"[PrinterInterface] ERROR resuming via OctoPrint: {e}")
            return False

    def _octoprint_cancel(self) -> bool:
        try:
            import requests

            r = requests.post(
                f"{self.api_url}/api/job",
                json={"command": "cancel"},
                headers=self._octoprint_headers(),
                timeout=_HTTP_TIMEOUT,
            )
            ok = r.status_code == 204
            print(f"[PrinterInterface] OctoPrint cancel -> {r.status_code}")
            return ok
        except Exception as e:
            print(f"[PrinterInterface] ERROR cancelling via OctoPrint: {e}")
            return False


if __name__ == "__main__":
    # Quick connectivity self-test:
    #   PRINTER_MODE=live PRINTER_URL=http://<giga-ip>:7125 python -m src.printer_interface
    p = PrinterInterface()
    print("ping       ->", p.ping())
    print("is_printing->", p.is_printing())
