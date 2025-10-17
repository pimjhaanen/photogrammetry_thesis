#!/usr/bin/env python3
"""
... (same header as your version) ...
"""

from pypozyx import (
    PozyxSerial, POZYX_SUCCESS, get_first_pozyx_serial_port, DeviceRange
)

try:
    from pypozyx.structures.device import (
        UWBSettings,
        UWB_CHANNEL_5, UWB_CHANNEL_9,
        UWB_PRF_64MHZ, UWB_PLEN_256, UWB_BITRATE_6M8
    )
    _HAS_UWB_CONSTS = True
except Exception:
    _HAS_UWB_CONSTS = False

import time, os, sys, csv, json
from typing import Optional, Tuple, List
import numpy as np
from datetime import datetime, timezone

# GPIO guarded import
try:
    import RPi.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False

# stdout line-buffered
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# -----------------
# Simple logger
# -----------------
_LOGFILE = None
def _log_open(path: str):
    global _LOGFILE
    _LOGFILE = open(path, "a", encoding="utf-8")
    _log(f"📄 Logging to {path}")

def _log(msg: str):
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    if _LOGFILE:
        _LOGFILE.write(line + "\n")
        _LOGFILE.flush()
        try: os.fsync(_LOGFILE.fileno())
        except Exception: pass

# -----------------
# Network UTC (one-shot) → offset correction
# -----------------
UTC_CORR = 0.0  # seconds; added to time.time() to get corrected UTC

def get_network_utc(timeout=3.0) -> Optional[float]:
    """Return epoch seconds from HTTP 'Date' header (UTC) if internet available."""
    try:
        import urllib.request, email.utils, calendar
        urls = ["https://google.com", "https://www.cloudflare.com", "https://www.microsoft.com"]
        for url in urls:
            try:
                req = urllib.request.Request(url, method="HEAD")
            except TypeError:
                req = urllib.request.Request(url)  # py<3.9 fallback (GET)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                date_hdr = resp.headers.get("Date")
                if not date_hdr:
                    continue
                dt = email.utils.parsedate_to_datetime(date_hdr)  # timezone-aware UTC
                return dt.timestamp()
    except Exception as e:
        _log(f"ℹ️ Network UTC fetch failed: {e}")
    return None

def init_utc_correction():
    global UTC_CORR
    net = get_network_utc()
    if net is None:
        _log("ℹ️ No network UTC available; using system time as-is.")
        UTC_CORR = 0.0
    else:
        sys_now = time.time()
        UTC_CORR = float(net - sys_now)
        sign = "+" if UTC_CORR >= 0 else "-"
        _log(f"⏱️ Applied UTC correction: {sign}{abs(UTC_CORR):.3f}s (net: {net:.3f}, sys: {sys_now:.3f})")

def now_epoch() -> float:
    """Corrected UTC epoch seconds (stable even after hotspot disconnect)."""
    return time.time() + UTC_CORR

# -----------------
# GPIO / Sync flash
# -----------------
def setup_gpio(pin: int = 14) -> None:
    if not _HAS_GPIO: return
    GPIO.setmode(GPIO.BCM); GPIO.setup(pin, GPIO.OUT); GPIO.output(pin, GPIO.LOW)

def flash_sync(duration_ms: int = 100, pin: int = 14) -> float:
    if _HAS_GPIO: GPIO.output(pin, GPIO.HIGH)
    t0_epoch = now_epoch()
    _log(f"⚡ Sync LED ON @ {datetime.fromtimestamp(t0_epoch, tz=timezone.utc).isoformat()}")
    time.sleep(duration_ms / 1000.0)
    if _HAS_GPIO: GPIO.output(pin, GPIO.LOW)
    return t0_epoch

def cleanup_gpio() -> None:
    if _HAS_GPIO: GPIO.cleanup()

# -----------------
# Utilities
# -----------------
def utc_iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat()

def print_error(pozyx, who: Optional[int] = None) -> None:
    try:
        if who is not None:
            err = pozyx.getErrorCode(who); _log(f"❗ Pozyx error @0x{who:04X}: {err}")
        else:
            err = pozyx.getErrorCode(); _log(f"❗ Pozyx error (local): {err}")
    except Exception as e:
        _log(f"❗ Could not read error code: {e}")

def configure_radios(pozyx, channel: Optional[int],
                     remote_id: Optional[int],
                     destination_id: int,
                     tx_power_remote: int,
                     tx_power_destination: int) -> None:
    if not _HAS_UWB_CONSTS or channel is None:
        _log("ℹ️ Skipping UWB reconfig (constants unavailable or no channel requested).")
        return
    try:
        uwb = UWBSettings(channel=channel, prf=UWB_PRF_64MHZ,
                          plen=UWB_PLEN_256, bitrate=UWB_BITRATE_6M8)
        pozyx.setUWBSettings(uwb)
        pozyx.setUWBSettings(uwb, destination_id)
        if remote_id is not None: pozyx.setUWBSettings(uwb, remote_id)
        try: pozyx.setTxPower(tx_power_destination, destination_id)
        except Exception as e: _log(f"⚠️ setTxPower(destination) failed: {e}")
        if remote_id is not None:
            try: pozyx.setTxPower(tx_power_remote, remote_id)
            except Exception as e: _log(f"⚠️ setTxPower(remote) failed: {e}")
        _log(f"🔧 UWB settings applied on channel {channel}.")
    except Exception as e:
        _log(f"⚠️ configure_radios failed: {e}")

def open_pozyx_or_die() -> Tuple[PozyxSerial, DeviceRange, str]:
    port = get_first_pozyx_serial_port()
    if port is None:
        _log("❌ No Pozyx USB device found.")
        raise RuntimeError("No Pozyx serial port")
    _log(f"🔌 Opening Pozyx on port: {port}")
    return PozyxSerial(port), DeviceRange(), port

# -----------------
# Post-processing (unchanged except using utc_iso)
# -----------------
def apply_postprocessing(raw_csv_path: str,
    calibration_path: Optional[str] = "calibration/uwb_calibration.json",
    apply_low_pass: bool = True, alpha: float = 0.95) -> str:
    """
    Post-process a RAW UWB CSV:
      - calibrate y = a*x + b (from JSON if present)
      - interpolate over NaNs
      - zero-phase EMA smoothing (forward + backward) if apply_low_pass
      - write processed CSV with same columns as before

    Zero-phase EMA removes lag:
        fwd[t] = α fwd[t-1] + (1-α) x[t]
        bwd = reverse( EMA( reverse(fwd) ) )
    """

    def _ema_pass(x: np.ndarray, a: float) -> np.ndarray:
        """One-direction EMA with NaN carry; starts at first finite sample."""
        y = np.empty_like(x, dtype=float)
        y[:] = np.nan
        idx = np.where(np.isfinite(x))[0]
        if len(idx) == 0:
            return y
        i0 = idx[0]
        y[i0] = x[i0]
        for i in range(i0 + 1, len(x)):
            xi = x[i]
            y[i] = (a * y[i - 1] + (1.0 - a) * xi) if np.isfinite(xi) else y[i - 1]
        return y

    def _zero_phase_ema(x: np.ndarray, a: float) -> np.ndarray:
        if x.size == 0:
            return x
        fwd = _ema_pass(x, a)
        bwd = _ema_pass(fwd[::-1], a)[::-1]
        return bwd

    # --- Load calibration (a, b) ---
    a, b = 1.0, 0.0
    if calibration_path and os.path.isfile(calibration_path):
        with open(calibration_path, "r") as f:
            calib = json.load(f)
        a = float(calib.get("a", 1.0))
        b = float(calib.get("b", 0.0))
        _log(f"🧪 Using calibration: y = {a:.6f} * x + {b:.6f}")
    else:
        _log("ℹ️ No calibration file found — using identity (a=1, b=0).")

    # --- Read RAW ---
    epochs: List[float] = []
    distances_raw: List[float] = []
    with open(raw_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("Timestamp (s)")
            d = row.get("Distance (m)")
            if not t:
                continue
            epochs.append(float(t))
            distances_raw.append(float(d) if d not in (None, "") else np.nan)

    if not epochs:
        raise ValueError("No samples found in raw CSV.")

    # --- Time base & UTC strings ---
    start = epochs[0]
    elapsed_s = [ts - start for ts in epochs]
    utc_strs = [utc_iso(ts) for ts in epochs]

    # --- Calibrate & fill gaps ---
    y = np.array([a * val + b if not np.isnan(val) else np.nan for val in distances_raw], dtype=float)
    x_idx = np.arange(len(y))
    mask = np.isfinite(y)
    if mask.sum() >= 2:
        y_interp = np.interp(x_idx, x_idx[mask], y[mask])
    else:
        y_interp = y.copy()

    # Keep provenance
    sources = ["reality" if np.isfinite(orig) else "interpolated" for orig in y]

    # --- Smoothing (zero-phase EMA) ---
    if apply_low_pass and y_interp.size > 0:
        y_out = _zero_phase_ema(y_interp, alpha)
    else:
        y_out = y_interp

    # --- Write processed CSV ---
    out = raw_csv_path.replace("_raw.csv", ".csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["UTC (ISO8601)", "Timestamp (s)", "Distance (m)", "Source"])
        for u, rel, d, s in zip(utc_strs, elapsed_s, y_out, sources):
            w.writerow([u, f"{rel:.6f}", "" if np.isnan(d) else f"{float(d):.6f}", s])

    _log(f"✅ Processed data written to: {os.path.abspath(out)}")
    return out
# -----------------
# UWB logger (only timestamps now use now_epoch())
# -----------------
def run_uwb_logger(
    destination_id: int,
    remote_id: Optional[int] = None,
    use_remote_initiator: bool = False,
    tx_power_remote: int = 33,
    tx_power_destination: int = 33,
    sample_dt_s: float = 0.05,
    output_dir: str = "output",
    calibration_path: Optional[str] = "calibration/uwb_calibration.json",
    gpio_pin: int = 14,
    flash_ms: int = 100,
    do_postprocess: bool = True,
    soft_reset_after_misses: int = 40,
    reopen_after_misses: int = 120,
    channels_cycle: Optional[List[int]] = None
) -> Tuple[str, Optional[str]]:

    # One-shot network UTC correction (if hotspot reachable)
    init_utc_correction()

    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now(timezone.utc); timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    side_log_path = f"/tmp/uwb_{timestamp_str}.log"; _log_open(side_log_path)
    base = f"uwb_flight_{timestamp_str}"; raw_csv_path = os.path.join(output_dir, base + "_raw.csv")

    pozyx, device_range, port = open_pozyx_or_die()

    if _HAS_UWB_CONSTS:
        if channels_cycle is None: channels_cycle = [UWB_CHANNEL_5, UWB_CHANNEL_9]
        chan_index = 0; current_channel = channels_cycle[chan_index]
    else:
        chan_index = 0; current_channel = None

    configure_radios(pozyx, current_channel, remote_id, destination_id, tx_power_remote, tx_power_destination)
    setup_gpio(gpio_pin)

    post_path = None
    with open(raw_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["UTC (ISO8601)", "Timestamp (s)", "Distance (m)", "Status"])
        f.flush(); os.fsync(f.fileno())

        try:
            # --- keep just the flash row as the first timestamp ---
            t0_epoch = flash_sync(duration_ms=flash_ms, pin=gpio_pin)  # returns epoch at LED HIGH
            writer.writerow([utc_iso(t0_epoch), f"{t0_epoch:.6f}", "", "flash"])
            f.flush();
            os.fsync(f.fileno())

            _log(f"✅ RAW opened and flash marker written: {os.path.abspath(raw_csv_path)}")
            _log("▶️ Starting UWB ranging loop (Ctrl+C to stop)...")

            misses = 0
            while True:
                tick_t0 = time.monotonic()
                try:
                    # --- choose the right ranging call for your pypozyx version ---
                    if use_remote_initiator and (remote_id is not None):
                        # Ask the REMOTE device (0x%04X) to range TO destination (0x%04X)
                        status = pozyx.doRanging(destination_id, device_range, remote_id)
                    else:
                        # Local USB device ranges TO destination
                        status = pozyx.doRanging(destination_id, device_range)

                    now_e = now_epoch()  # <-- corrected UTC
                    if status == POZYX_SUCCESS:
                        dist_m = device_range.distance / 1000.0
                        writer.writerow([utc_iso(now_e), f"{now_e:.6f}", f"{float(dist_m):.6f}", "ok"])
                        _log(f"📏 {dist_m:.3f} m")
                        misses = 0
                    else:
                        writer.writerow([utc_iso(now_e), f"{now_e:.6f}", "", "fail"])
                        _log("⚠️ Ranging failed")
                        print_error(pozyx, remote_id if use_remote_initiator and remote_id is not None else None)
                        misses += 1

                        if misses == soft_reset_after_misses:
                            _log("🔧 Soft reset + re-apply UWB (sustained fails)")
                            try: pozyx.resetSystem(); time.sleep(0.5)
                            except Exception as e: _log(f"⚠️ resetSystem failed: {e}")
                            if _HAS_UWB_CONSTS and channels_cycle:
                                chan_index = (chan_index + 1) % len(channels_cycle)
                                current_channel = channels_cycle[chan_index]
                            else:
                                current_channel = None
                            configure_radios(pozyx, current_channel, remote_id, destination_id, tx_power_remote, tx_power_destination)

                        if misses >= reopen_after_misses:
                            _log("🧰 Hard recovery: re-open serial and reconfigure")
                            try:
                                probe = get_first_pozyx_serial_port()
                                if probe is None:
                                    _log("❌ Pozyx USB device not found (cable disconnected) — stopping.")
                                    break
                                try: del pozyx
                                except Exception: pass
                                time.sleep(0.2)
                                pozyx, device_range, port = open_pozyx_or_die()
                                configure_radios(pozyx, current_channel, remote_id, destination_id, tx_power_remote, tx_power_destination)
                                misses = 0
                            except Exception as e:
                                _log(f"❌ Re-open failed: {e}")

                    f.flush(); os.fsync(f.fileno())

                    elapsed = time.monotonic() - tick_t0
                    sleep_left = sample_dt_s - elapsed
                    if sleep_left > 0: time.sleep(sleep_left)

                except KeyboardInterrupt:
                    _log("⏹️ Interrupted by user (Ctrl+C). Saving & exit.")
                    break

                except Exception as e:
                    _log(f"❌ Exception in loop: {e}")
                    now_e = now_epoch()
                    writer.writerow([utc_iso(now_e), f"{now_e:.6f}", "", f"exception:{e.__class__.__name__}"])
                    f.flush(); os.fsync(f.fileno())
                    probe = get_first_pozyx_serial_port()
                    if probe is None:
                        _log("❌ Pozyx USB device missing — stopping.")
                        break
                    else:
                        _log("🧰 Attempting immediate re-open after exception...")
                        try: del pozyx
                        except Exception: pass
                        time.sleep(0.2)
                        pozyx, device_range, port = open_pozyx_or_die()
                        configure_radios(pozyx, current_channel, remote_id, destination_id, tx_power_remote, tx_power_destination)
                        misses = 0
                        continue
        finally:
            try: f.flush(); os.fsync(f.fileno())
            except Exception: pass

    if do_postprocess:
        try:
            post_path = apply_postprocessing(os.path.abspath(raw_csv_path),
                                             calibration_path=calibration_path,
                                             apply_low_pass=True, alpha=0.95)
        except Exception as e:
            _log(f"❗ Post-processing failed: {e}")
            post_path = None

    cleanup_gpio()
    try:
        if _LOGFILE: _log("🧹 Closing side log."); _LOGFILE.close()
    except Exception: pass

    return raw_csv_path, post_path

# -------------
# Script entry
# -------------
if __name__ == "__main__":
    DESTINATION_ID = 0x6931   # anchor B
    REMOTE_ID      = 0x6923   # anchor A (the initiator)

    RAW_PATH, PROC_PATH = run_uwb_logger(
        destination_id=DESTINATION_ID,
        remote_id=REMOTE_ID,
        use_remote_initiator=True,      # <<< MEASURE 0x6923 ↔ 0x6931
        tx_power_remote=33, tx_power_destination=33,
        sample_dt_s=0.05,
        output_dir="output",
        calibration_path="calibration/uwb_calibration.json",
        gpio_pin=14, flash_ms=100, do_postprocess=True,
        soft_reset_after_misses=40, reopen_after_misses=120,
        channels_cycle=[UWB_CHANNEL_5, UWB_CHANNEL_9] if _HAS_UWB_CONSTS else None
    )
    print(f"RAW CSV: {RAW_PATH}")
    print(f"PROC CSV: {PROC_PATH}")
