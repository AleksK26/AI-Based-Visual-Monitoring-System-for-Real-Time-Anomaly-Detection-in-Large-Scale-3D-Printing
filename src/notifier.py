"""
Alert dispatch for confirmed defect events.

When the detector confirms a defect (a class crosses the temporal min-hits), the
monitoring loop calls Notifier.notify_defect(...). This module fans that event out
to one or more channels (Telegram, email) and, optionally, attaches the annotated
snapshot so you can eyeball the defect from your phone without being at the machine.

Design goals:
  * NEVER crash the monitoring loop — every send is wrapped in try/except and uses a
    short network timeout, so a flaky connection can't stall detection.
  * No spam — a per-class cooldown suppresses repeat alerts while a defect persists.
  * Zero-config safe — with no env vars set, alerts are just logged to stdout, so the
    system runs anywhere out of the box.

Configuration (all via environment variables, all optional):

    NOTIFY_CHANNELS     comma list of channels to use: "telegram,email"
                        (default: "" -> console log only)
    NOTIFY_COOLDOWN     seconds to suppress repeat alerts for the same class
                        (default: 120)

    # --- Telegram (easiest: create a bot with @BotFather, message it once) ---
    TELEGRAM_BOT_TOKEN  token from @BotFather
    TELEGRAM_CHAT_ID    your chat id (message @userinfobot to get it)

    # --- Email (SMTP, e.g. Gmail with an App Password) ---
    SMTP_HOST           e.g. smtp.gmail.com
    SMTP_PORT           default 587 (STARTTLS)
    SMTP_USER           login / "from" address
    SMTP_PASSWORD       password or app-password
    NOTIFY_EMAIL_TO     recipient (default: SMTP_USER)

Example:
    NOTIFY_CHANNELS=telegram TELEGRAM_BOT_TOKEN=123:abc TELEGRAM_CHAT_ID=456 \
        streamlit run app.py
"""

import os
import time

# Short timeout so a slow/unreachable notification endpoint never stalls the loop.
_HTTP_TIMEOUT = 5


def _encode_jpeg(frame_bgr):
    """Encode a BGR frame (numpy array) to JPEG bytes. Returns None on any failure."""
    if frame_bgr is None:
        return None
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None
    except Exception as e:
        print(f"[Notifier] Could not encode snapshot: {e}")
        return None


class Notifier:
    """Fans confirmed-defect events out to the configured alert channels."""

    def __init__(self):
        channels = os.environ.get("NOTIFY_CHANNELS", "")
        self.channels = [c.strip().lower() for c in channels.split(",") if c.strip()]
        self.cooldown = float(os.environ.get("NOTIFY_COOLDOWN", "120"))

        # Telegram
        self.tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

        # Email / SMTP
        self.smtp_host = os.environ.get("SMTP_HOST", "")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("SMTP_USER", "")
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self.email_to = os.environ.get("NOTIFY_EMAIL_TO", "") or self.smtp_user

        # Per-class timestamp of the last alert sent, for cooldown suppression.
        self._last_sent: dict[str, float] = {}

        if self.channels:
            print(f"[Notifier] Channels: {', '.join(self.channels)} "
                  f"(cooldown {self.cooldown:.0f}s)")
        else:
            print("[Notifier] No channels configured - defect alerts log to console only.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_defect(self, classes, frame_bgr=None, paused=False) -> bool:
        """
        Announce a confirmed defect.

        Args:
            classes:   list[str] of confirmed defect class names.
            frame_bgr: optional annotated BGR frame to attach as a snapshot.
            paused:    True if the printer was auto-paused as a result.

        Returns:
            True if at least one alert was dispatched (or logged); False if the
            event was fully suppressed by the cooldown.
        """
        classes = list(classes)
        if not classes:
            return False

        # Cooldown: only alert for classes not alerted within the cooldown window.
        now = time.time()
        fresh = [c for c in classes if now - self._last_sent.get(c, 0.0) >= self.cooldown]
        if not fresh:
            return False
        for c in fresh:
            self._last_sent[c] = now

        action = "Printer AUTO-PAUSED" if paused else "Detected (no auto-pause)"
        title = "🛑 3D Print Defect Confirmed"
        body = (f"{title}\n"
                f"Defect(s): {', '.join(fresh)}\n"
                f"Action: {action}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Always log — this is the zero-config fallback and an on-host audit trail.
        # Sanitise to ASCII so a cp1252 Windows console can't raise on the emoji.
        log_line = body.replace(chr(10), " | ").encode("ascii", "replace").decode("ascii")
        print(f"[Notifier] {log_line}")

        jpeg = _encode_jpeg(frame_bgr)
        for ch in self.channels:
            try:
                if ch == "telegram":
                    self._send_telegram(body, jpeg)
                elif ch == "email":
                    self._send_email(title, body, jpeg)
                else:
                    print(f"[Notifier] Unknown channel '{ch}', skipping.")
            except Exception as e:
                # A failing channel must never break monitoring.
                print(f"[Notifier] ERROR sending via {ch}: {e}")
        return True

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------

    def _send_telegram(self, text, jpeg):
        if not (self.tg_token and self.tg_chat_id):
            print("[Notifier] Telegram not configured (need TELEGRAM_BOT_TOKEN + "
                  "TELEGRAM_CHAT_ID).")
            return
        import requests

        base = f"https://api.telegram.org/bot{self.tg_token}"
        if jpeg:
            r = requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": self.tg_chat_id, "caption": text},
                files={"photo": ("defect.jpg", jpeg, "image/jpeg")},
                timeout=_HTTP_TIMEOUT,
            )
        else:
            r = requests.post(
                f"{base}/sendMessage",
                data={"chat_id": self.tg_chat_id, "text": text},
                timeout=_HTTP_TIMEOUT,
            )
        print(f"[Notifier] Telegram -> {r.status_code}")

    # ------------------------------------------------------------------
    # Email (SMTP)
    # ------------------------------------------------------------------

    def _send_email(self, subject, body, jpeg):
        if not (self.smtp_host and self.smtp_user and self.email_to):
            print("[Notifier] Email not configured (need SMTP_HOST, SMTP_USER, "
                  "NOTIFY_EMAIL_TO).")
            return
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = self.email_to
        msg.set_content(body)
        if jpeg:
            msg.add_attachment(jpeg, maintype="image", subtype="jpeg",
                               filename="defect.jpg")

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=_HTTP_TIMEOUT) as s:
            s.starttls()
            if self.smtp_password:
                s.login(self.smtp_user, self.smtp_password)
            s.send_message(msg)
        print(f"[Notifier] Email -> sent to {self.email_to}")


if __name__ == "__main__":
    # Quick self-test:
    #   NOTIFY_CHANNELS=telegram TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
    #       python -m src.notifier
    n = Notifier()
    n.notify_defect(["Spaghetti"], paused=True)
