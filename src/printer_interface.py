"""
The abstraction layer for printer communication.
CURRENT ROLE: Mock mode returns safe status codes so the full pipeline
can be tested without a physical printer.
FUTURE ROLE: Will send actual HTTP/API requests to the printer firmware
(OctoPrint or Moonraker) to check status and pause the print on defect.
"""

import os

# Set PRINTER_MODE=live in your environment to enable real HTTP calls.
# Leave unset (or set to 'mock') for safe testing without a printer.
PRINTER_MODE = os.environ.get("PRINTER_MODE", "mock").lower()


class PrinterInterface:
    """
    Unified interface for communicating with the printer firmware.

    Mock mode: logs actions to stdout, safe to run anywhere.
    Live mode (OctoPrint / Moonraker): sends HTTP API requests.
    """

    def __init__(self, api_url: str = "", api_key: str = ""):
        """
        Args:
            api_url: Base URL of OctoPrint or Moonraker
                     e.g. 'http://octopi.local' or 'http://192.168.1.x'
            api_key: OctoPrint API key (not required for Moonraker).
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self._mode = PRINTER_MODE
        print(f"[PrinterInterface] Mode: {self._mode.upper()}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_printing(self) -> bool:
        """Returns True if the printer is currently printing."""
        if self._mode == "mock":
            print("[PrinterInterface] MOCK: is_printing() → True")
            return True
        return self._live_is_printing()

    def pause_print(self) -> bool:
        """
        Pause the active print job.

        Returns:
            True if the pause command was accepted, False otherwise.
        """
        if self._mode == "mock":
            print("[PrinterInterface] MOCK: pause_print() → paused")
            return True
        return self._live_pause()

    # ------------------------------------------------------------------
    # Live implementations (fill in when ready for real hardware)
    # ------------------------------------------------------------------

    def _live_is_printing(self) -> bool:
        """
        Query OctoPrint or Moonraker for current print state.
        Uncomment the relevant block for your firmware.
        """
        try:
            import requests

            # --- OctoPrint ---
            # r = requests.get(
            #     f"{self.api_url}/api/job",
            #     headers={"X-Api-Key": self.api_key},
            #     timeout=3,
            # )
            # return r.json().get("state", "") == "Printing"

            # --- Moonraker ---
            # r = requests.get(f"{self.api_url}/printer/objects/query?print_stats", timeout=3)
            # return r.json()["result"]["status"]["print_stats"]["state"] == "printing"

            raise NotImplementedError("Configure live firmware URL/key above.")

        except Exception as e:
            print(f"[PrinterInterface] ERROR querying printer: {e}")
            return False

    def _live_pause(self) -> bool:
        """
        Send pause command to OctoPrint or Moonraker.
        Uncomment the relevant block for your firmware.
        """
        try:
            import requests

            # --- OctoPrint ---
            # r = requests.post(
            #     f"{self.api_url}/api/job",
            #     json={"command": "pause", "action": "pause"},
            #     headers={"X-Api-Key": self.api_key},
            #     timeout=3,
            # )
            # return r.status_code == 204

            # --- Moonraker ---
            # r = requests.post(f"{self.api_url}/printer/print/pause", timeout=3)
            # return r.status_code == 200

            raise NotImplementedError("Configure live firmware URL/key above.")

        except Exception as e:
            print(f"[PrinterInterface] ERROR pausing printer: {e}")
            return False
