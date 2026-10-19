"""
出席コード監視ツール (macOS / MacBook 完全最適化版 - AppleScript直接入力＆B/8自動補正＆完全即時停止対応)
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import threading
import time
import warnings
import webbrowser
import tkinter as tk
from tkinter import ttk
from collections import Counter, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ==========================================================
# macOS 専用設定
# ==========================================================
MOD_KEY = "command"

# ==========================================================
# 基本設定
# ==========================================================
CHECK_INTERVAL = 0.15
VERIFY_WINDOW = 4.0
VERIFY_INTERVAL = 0.8
RECHECK_INTERVAL = 5.0
ABSENT_RESET = 6.0
COOLDOWN_SEC = 600
FAIL_SKIP_SEC = 3.0
FULLSCAN_INTERVAL = 1.0
FULLSCAN_WIDTH = 1280

DETECT_WIDTH = 960
HSV_RED = (((0, 100, 120), (14, 255, 255)), ((172, 100, 120), (179, 255, 255)))
BOX_MIN = (16, 8)
BOX_ASPECT = (1.2, 3.5)
BOX_MIN_FILL = 0.80
BOX_MIN_WHITE = 0.02
MAX_BOXES = 3

PANEL_HALF_WIDTH = 2.6
PANEL_BELOW = 5.0
OCR_TARGET_WIDTH = 1000

CODE_LEN = 6
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CODE_RE = re.compile(rf"^[A-Z0-9]{{{CODE_LEN}}}$")

SAVE_EVIDENCE = True
SAVE_DIR = Path("attendance_evidence")
LOG_FILE = "attendance_fast.log"

TAYLORS_REDIRECT_URL = "https://sdtaylorsedu.sharepoint.com/sites/CC/SitePages/Attendance-Marking.aspx"

_xcheck = {"enabled": True}
ui_app = None

# システム全体の緊急停止・中断イベント
pipeline_stop_event = threading.Event()


def interruptible_sleep(seconds: float) -> bool:
    """指定秒数待機する。途中で一時停止された場合は True を返して即終了する"""
    return pipeline_stop_event.wait(timeout=seconds)


def say(msg: str) -> None:
    line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if ui_app:
        ui_app.log_message(msg)


def enable_dpi_awareness() -> None:
    pass


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _span(parts: List[Line], text: str = "") -> Line:
    return Line(text, min(p.x0 for p in parts), min(p.y0 for p in parts),
                max(p.x1 for p in parts), max(p.y1 for p in parts))


def to_lines(results) -> List[Line]:
    items = []
    for bbox, text, _conf in results:
        t = str(text).strip()
        if not t:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append(Line(t, min(xs), min(ys), max(xs), max(ys)))
    items.sort(key=lambda ln: ln.cy)

    groups: List[List[Line]] = []
    for it in items:
        if groups:
            cur = _span(groups[-1])
            if abs(it.cy - cur.cy) < 0.5 * max(it.h, cur.h):
                groups[-1].append(it)
                continue
        groups.append([it])

    lines = []
    for g in groups:
        g.sort(key=lambda p: p.x0)
        lines.append(_span(g, " ".join(p.text for p in g)))
    return lines


def _letters(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


_REMAIN_NUM = re.compile(r"\d+\s*m\w*\W*\d+\s*s", re.I)


def is_remaining(text: str) -> bool:
    letters = _letters(text)
    if len(letters) >= 6 and SequenceMatcher(None, letters[:9], "REMAINING").ratio() >= 0.75:
        return True
    return bool(_REMAIN_NUM.search(text))


def find_anchor(lines: List[Line]) -> Optional[Line]:
    best, best_r = None, 0.0
    for ln in lines:
        letters = _letters(ln.text)
        if not 5 <= len(letters) <= 12:
            continue
        r = SequenceMatcher(None, letters, "PASSCODE").ratio()
        if r > best_r:
            best, best_r = ln, r
    return best if best_r >= 0.8 else None


def _red_mask(hsv):
    mask = None
    for lo, hi in HSV_RED:
        part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = part if mask is None else (mask | part)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def find_stop_boxes(frame_bgr, max_boxes: int = MAX_BOXES, return_mask: bool = False):
    height, width = frame_bgr.shape[:2]
    scale = min(1.0, DETECT_WIDTH / float(width))
    small = frame_bgr if scale >= 1.0 else cv2.resize(
        frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = _red_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < BOX_MIN[0] or h < BOX_MIN[1]:
            continue
        if not (BOX_ASPECT[0] <= w / h <= BOX_ASPECT[1]):
            continue
        if cv2.contourArea(c) / float(w * h) < BOX_MIN_FILL:
            continue
        region = hsv[y:y + h, x:x + w]
        white = (region[:, :, 1] < 80) & (region[:, :, 2] > 180)
        if white.mean() < BOX_MIN_WHITE:
            continue
        found.append((w * h, (x, y, w, h)))
    found.sort(key=lambda t: -t[0])
    boxes = [tuple(int(round(v / scale)) for v in b) for _, b in found[:max_boxes]]
    return (boxes, mask) if return_mask else boxes


def crop_panel(frame, box):
    height, width = frame.shape[:2]
    x, y, w, h = box
    cx = x + w / 2.0
    x0 = max(0, int(cx - PANEL_HALF_WIDTH * w))
    x1 = min(width, int(cx + PANEL_HALF_WIDTH * w))
    y0 = max(0, int(y + h))
    y1 = min(height, int(y + h * PANEL_BELOW))
    return frame[y0:y1, x0:x1], x0, y0


@dataclass
class Panel:
    code: Optional[str]
    primary: Optional[str]
    code_line: Line
    anchor: Line
    remaining: str
    stop_ok: bool = True


def validate_panel(lines: List[Line], ref_cx: Optional[float] = None,
                   ref_w: Optional[float] = None) -> Tuple[Optional[Panel], str]:
    anchor = find_anchor(lines)
    if anchor is None:
        return None, "「Passcode」の文字を確認できません"
    if ref_cx is not None and abs(anchor.cx - ref_cx) > 0.9 * ref_w:
        return None, "Passcodeが赤枠の真下にありません"

    cands = []
    for ln in lines:
        if ln is anchor or ln.cy <= anchor.cy or is_remaining(ln.text):
            continue
        if not 4 <= len(_alnum(ln.text)) <= 9:
            continue
        if ln.h < anchor.h * 1.2:
            continue
        if ln.cy - anchor.cy > 5 * anchor.h:
            continue
        if abs(ln.cx - anchor.cx) > 0.45 * max(ln.w, anchor.w):
            continue
        if ref_cx is not None and abs(ln.cx - ref_cx) > 0.9 * ref_w:
            continue
        cands.append(ln)
    if not cands:
        return None, "Passcodeの下に、それより大きい6文字前後の行がありません"
    code_line = min(cands, key=lambda ln: ln.cy)

    remaining = None
    for ln in lines:
        if ln.cy > code_line.cy and ln.cy - code_line.cy <= 3 * code_line.h and is_remaining(ln.text):
            remaining = ln
            break
    if remaining is None:
        return None, "コードの下に「Remaining」の行がありません"
    if abs(remaining.cx - code_line.cx) > 0.4 * remaining.w:
        return None, "Remainingの行がコードの真下に揃っていません"

    compact = _alnum(code_line.text)
    primary = compact if CODE_RE.match(compact) else None
    return Panel(None, primary, code_line, anchor, remaining.text), "OK"


def refine_code(reader, img_bgr, line: Line, primary: Optional[str]) -> Optional[str]:
    if not _xcheck["enabled"]:
        return primary
    height, width = img_bgr.shape[:2]
    pad = int(line.h * 0.2) + 2
    x0, x1 = max(0, int(line.x0) - pad), min(width, int(line.x1) + pad)
    y0, y1 = max(0, int(line.y0) - pad), min(height, int(line.y1) + pad)
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return primary
    k = float(np.clip(96.0 / crop.shape[0], 0.5, 3.0))
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def variants():
        yield gray
        yield cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield otsu if otsu.mean() >= 127 else 255 - otsu

    support, weight = Counter(), Counter()
    if primary:
        support[primary] += 1
    for v in variants():
        try:
            res = reader.recognize(v, allowlist=ALPHABET, detail=1)
        except Exception as e:
            if _xcheck["enabled"]:
                _xcheck["enabled"] = False
                say(f"[注意] 再読による照合が使えないため、一次読みのみで判定します({e})")
            break
        if not res:
            continue
        text, conf = _alnum(res[0][1]), float(res[0][2])
        if not CODE_RE.match(text):
            continue
        if primary and text == primary:
            return primary
        support[text] += 1
        weight[text] += conf
    if not support:
        return None
    best, n = max(support.items(), key=lambda kv: (kv[1], kv[0] == primary, weight[kv[0]]))
    if primary is None and n < 2:
        return None
    return best


# ==========================================================
# 【B と 8 の誤読完全防止】幾何学補正モジュール
# ==========================================================
def _disambiguate_char(char_gray: np.ndarray) -> str:
    if char_gray.size == 0 or min(char_gray.shape) < 5:
        return 'B'
    if char_gray.mean() > 127:
        _, binary = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    if len(rows) < 5 or len(cols) < 5:
        return 'B'
    
    roi = binary[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    h, w = roi.shape
    
    mid_y0, mid_y1 = int(h * 0.15), int(h * 0.85)
    left_x = []
    for y in range(mid_y0, mid_y1):
        x_idx = np.where(roi[y, :] > 0)[0]
        if len(x_idx) > 0:
            left_x.append(x_idx[0])
            
    if not left_x:
        return 'B'
        
    x_range = np.max(left_x) - np.min(left_x)
    norm_range = x_range / float(w)
    return 'B' if norm_range < 0.12 else '8'


def fix_B_and_8(code: str, roi_img: np.ndarray, line: Line) -> str:
    if not code or len(code) != 6 or not any(c in ('B', '8') for c in code):
        return code
    try:
        h_img, w_img = roi_img.shape[:2]
        pad = int(line.h * 0.2) + 2
        x0, x1 = max(0, int(line.x0) - pad), min(w_img, int(line.x1) + pad)
        y0, y1 = max(0, int(line.y0) - pad), min(h_img, int(line.y1) + pad)
        crop = roi_img[y0:y1, x0:x1]
        if crop.size == 0:
            return code
            
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        if gray.mean() > 127:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        if len(rows) < 5 or len(cols) < 5:
            return code
            
        tight_gray = gray[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
        tight_bin = binary[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
        th, tw = tight_gray.shape
        
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(tight_bin, connectivity=8)
        boxes = []
        for idx in range(1, n_labels):
            x, y, cw, ch, area = stats[idx]
            if ch > th * 0.4 and area > 20:
                boxes.append((x, y, cw, ch))
        boxes.sort(key=lambda b: b[0])
        
        char_crops = []
        if len(boxes) == 6:
            for x, y, cw, ch in boxes:
                char_crops.append(tight_gray[y:y+ch, x:x+cw])
        else:
            step = tw / 6.0
            for i in range(6):
                char_crops.append(tight_gray[:, int(i*step):int((i+1)*step)])
                
        code_list = list(code)
        for i, c in enumerate(code_list):
            if c in ('B', '8') and i < len(char_crops):
                correct = _disambiguate_char(char_crops[i])
                if correct != c:
                    say(f"[自動補正] {i+1}文字目を '{c}' ➔ '{correct}' に幾何学補正しました！")
                    code_list[i] = correct
                    
        return "".join(code_list)
    except Exception:
        return code


def read_panel(reader, frame, box):
    roi, ox, oy = crop_panel(frame, box)
    if roi.shape[0] < 20 or roi.shape[1] < 40:
        return None, "切り出し範囲が小さすぎます", []
    k = float(np.clip(OCR_TARGET_WIDTH / roi.shape[1], 1.0, 3.0))
    roi_up = cv2.resize(roi, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) if k > 1.01 else roi

    lines = to_lines(reader.readtext(roi_up))
    bx, by, bw, bh = box
    ref_cx = ((bx + bw / 2.0) - ox) * k
    panel, reason = validate_panel(lines, ref_cx, bw * k)
    if panel is None:
        return None, reason, lines
    code = refine_code(reader, roi_up, panel.code_line, panel.primary)
    if code is None:
        return None, "コード行は見つかりましたが、6文字として確定できません", lines
    code = fix_B_and_8(code, roi_up, panel.code_line)
    panel.code = code
    return panel, "OK", lines


class Tracker:
    def __init__(self):
        self.recent = {}
        self.reset()

    def reset(self):
        self.active = None
        self.pending = None
        self.notified_at = 0.0
        self.last_check = 0.0
        self.first_seen = None
        self.last_seen = 0.0

    def present(self, now: float):
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now

    def absent(self, now: float):
        if self.last_seen and now - self.last_seen > ABSENT_RESET:
            self.reset()

    def due(self, now: float) -> bool:
        if self.active is None:
            return True
        interval = VERIFY_INTERVAL if now - self.notified_at < VERIFY_WINDOW else RECHECK_INTERVAL
        return now - self.last_check >= interval

    def on_panel(self, code: str, now: float):
        self.last_check = now
        if self.active is None:
            self.active, self.pending = code, None
            seen = self.recent.get(code)
            self.recent[code] = now
            if seen is not None and now - seen < COOLDOWN_SEC:
                return None, None
            self.notified_at = now
            return "new", None
        if code == self.active:
            self.pending = None
            return None, None
        count = self.pending[1] + 1 if self.pending and self.pending[0] == code else 1
        self.pending = (code, count)
        if count < 2:
            return None, None
        prev = self.active
        kind = "fix" if len(code) == len(prev) and sum(x != y for x, y in zip(code, prev)) <= 2 else "new"
        self.active, self.pending = code, None
        self.recent[code] = now
        self.notified_at = now
        return kind, prev


class FailCache:
    def __init__(self):
        self.items = deque(maxlen=8)

    @staticmethod
    def _sig(frame, box):
        roi, _, _ = crop_panel(frame, box)
        if roi.size == 0:
            return None
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (32, 16), interpolation=cv2.INTER_AREA).astype(np.int16)

    def known(self, frame, box, now: float) -> bool:
        sig = self._sig(frame, box)
        if sig is None:
            return False
        for b, s, t in self.items:
            if now - t > FAIL_SKIP_SEC or s is None:
                continue
            if all(abs(b[i] - box[i]) <= 12 for i in range(3)) and np.abs(s - sig).mean() < 3.0:
                return True
        return False

    def add(self, frame, box, now: float):
        self.items.append((box, self._sig(frame, box), now))


def _beep():
    try:
        if not pipeline_stop_event.is_set():
            subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], check=False)
    except Exception:
        pass


def _toast(title: str, message: str):
    try:
        from plyer import notification
        notification.notify(title=title, message=message, app_name="出席コード監視", timeout=8)
    except Exception:
        try:
            safe_title = title.replace('"', '\\"')
            safe_msg = message.replace('"', '\\"')
            os.system(f'osascript -e \'display notification "{safe_msg}" with title "{safe_title}"\'')
        except Exception:
            pass


def _save_evidence(frame, code: str):
    try:
        SAVE_DIR.mkdir(exist_ok=True)
        path = SAVE_DIR / f"attendance_{datetime.datetime.now():%Y%m%d_%H%M%S}_{code}.png"
        cv2.imwrite(str(path), frame)
    except Exception as e:
        say(f"証拠画像の保存に失敗: {e}")


# ==========================================================
# macOS 専用: ブラウザ最前面化 (AppleScript)
# ==========================================================
def bring_browser_to_front() -> bool:
    if pipeline_stop_event.is_set():
        return False
    script = """
    tell application "System Events"
        set runningApps to name of every application process
    end tell
    repeat with appName in {"Safari", "Microsoft Edge", "Google Chrome", "Brave Browser"}
        if runningApps contains appName then
            tell application appName to activate
            return true
        end if
    end repeat
    return false
    """
    try:
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        time.sleep(0.25)
        return "true" in res.stdout
    except Exception:
        return False


# ==========================================================
# macOS 専用: AppleScript を使った直接タイプ入力
# ==========================================================
def mac_safe_type(text: str):
    if pipeline_stop_event.is_set():
        return
    import pyautogui
    try:
        script = f'''
        tell application "System Events"
            keystroke "a" using command down
            delay 0.1
            key code 51
            delay 0.1
            keystroke "{text}"
        end tell
        '''
        subprocess.run(["osascript", "-e", script], check=False)
    except Exception:
        if not pipeline_stop_event.is_set():
            pyautogui.write(text, interval=0.05)


# ==========================================================
# macOS 専用: リロード安定確認 ➔ パルスクリック ➔ 遷移確定検知
# ==========================================================
def wait_and_click_continue(reader, timeout: float = 50.0, poll_interval: float = 0.5) -> bool:
    import pyautogui
    import mss

    pyautogui.FAILSAFE = False
    deadline = time.monotonic() + timeout

    say(f"[Auto] リダイレクトページ待機中（リミット {timeout:.0f} 秒）...")

    consecutive_hits = 0
    last_bbox = None

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while time.monotonic() < deadline:
            if pipeline_stop_event.is_set():
                say("[Auto] システム一時停止を検知したため、待機処理を直ちに中断します。")
                return False

            bring_browser_to_front()

            frame = np.asarray(sct.grab(monitor))[:, :, :3]
            frame_h, frame_w = frame.shape[:2]

            crop_y0 = int(frame_h * 0.30)
            crop_y1 = int(frame_h * 0.80)
            roi = frame[crop_y0:crop_y1, :]

            results = reader.readtext(roi)
            target_bbox = None

            for bbox, text, conf in results:
                t_letters = _letters(text)
                if "CONTINUE" in t_letters or ("CLICK" in t_letters and "HERE" in t_letters):
                    target_bbox = bbox
                    break

            if target_bbox is not None:
                consecutive_hits += 1
                last_bbox = target_bbox

                if consecutive_hits < 2:
                    say("[Auto] リンクを発見。サイトのロード・リロード安定を待機中 (1/2)...")
                    if interruptible_sleep(0.8):
                        return False
                    continue

                say("[Auto] ★サイトのロード安定（静止状態）を確認！パルスクリックを開始します。")

                screen_w, screen_h = pyautogui.size()
                xs = [pt[0] for pt in last_bbox]
                ys = [pt[1] for pt in last_bbox]
                box_center_x = (min(xs) + max(xs)) / 2.0
                box_center_y = (min(ys) + max(ys)) / 2.0 + crop_y0

                click_x = round(screen_w * (box_center_x / float(frame_w)))
                click_y = round(screen_h * (box_center_y / float(frame_h)))

                for attempt in range(8):
                    if pipeline_stop_event.is_set():
                        return False

                    bring_browser_to_front()
                    say(f"[Auto] リンクをクリック送信 ({attempt + 1}/8)...")

                    if pipeline_stop_event.is_set():
                        return False
                    pyautogui.click(click_x, click_y)
                    if interruptible_sleep(0.15):
                        return False

                    if pipeline_stop_event.is_set():
                        return False
                    pyautogui.press('enter')
                    if interruptible_sleep(0.35):
                        return False

                    check_frame = np.asarray(sct.grab(monitor))[:, :, :3]
                    check_roi = check_frame[crop_y0:crop_y1, :]
                    check_results = reader.readtext(check_roi)
                    still_here = False
                    for _, c_text, _ in check_results:
                        if "CONTINUE" in _letters(c_text):
                            still_here = True
                            break

                    if not still_here:
                        say("★★★ [Auto] リダイレクト成功！出席画面へ完全に切り替わりました。★★★")
                        return True
                    else:
                        say("[Auto] まだサイトがリロード中または応答中です。次のクリックを送信します...")
                        if interruptible_sleep(0.2):
                            return False

                consecutive_hits = 0

            else:
                consecutive_hits = 0

            if interruptible_sleep(poll_interval):
                return False

    say(f"[Auto] {timeout:.0f}秒以内にリダイレクトを完了できませんでした。")
    return False


# ==========================================================
# macOS 専用: 自律提出パイプライン
# ==========================================================
def auto_submit_pipeline(code: str, reader):
    try:
        import pyautogui
        import mss

        if pipeline_stop_event.is_set():
            return

        say("[Auto] 出席自動提出プロセスを開始します...")

        say("[Auto] SharePointリダイレクトページを開きます...")
        webbrowser.open(TAYLORS_REDIRECT_URL)
        if interruptible_sleep(2.0):
            say("[Auto] 停止要求を検知したため、自動提出を中断しました。")
            return

        say("[Auto] リダイレクトページを監視中...（リミット50秒）")
        if not wait_and_click_continue(reader, timeout=50.0, poll_interval=0.5):
            say("[Auto] リダイレクト待機が終了または中断されました。")
            return

        if pipeline_stop_event.is_set():
            say("[Auto] 停止要求を検知したため、自動提出を中断しました。")
            return

        say("[Auto] 遷移先の出席入力画面を確認します。")
        submitted = False
        start_wait = time.time()

        with mss.mss() as sct:
            mon = sct.monitors[1]
            while time.time() - start_wait < 30.0:
                if pipeline_stop_event.is_set():
                    say("[Auto] 停止要求を検知したため、自動提出を中断しました。")
                    return

                bring_browser_to_front()

                cur_frame = np.array(sct.grab(mon))[:, :, :3]
                h, w = cur_frame.shape[:2]

                crop_y0, crop_y1 = int(h * 0.10), int(h * 0.50)
                search_roi = cur_frame[crop_y0:crop_y1, :]

                b_results = reader.readtext(search_roi)
                passcode_label = None
                button_bbox = None

                for bbox, text, _ in b_results:
                    t = text.strip().upper()
                    if "ENTER" in t or "PASSCODE" in t:
                        passcode_label = bbox

                    if "MARK" in t or "SUBMIT" in t:
                        button_bbox = bbox

                if passcode_label:
                    say("[Auto] ★本番の出席入力枠を発見しました！")
                    xs = [pt[0] for pt in passcode_label]
                    ys = [pt[1] for pt in passcode_label]
                    lx1 = max(xs)
                    ly = (min(ys) + max(ys)) / 2.0 + crop_y0

                    screen_w, screen_h = pyautogui.size()
                    input_x = round(screen_w * ((lx1 + 65) / float(w)))
                    input_y = round(screen_h * (ly / float(h)))

                    say(f"[Auto] 入力欄座標=({input_x}, {input_y}) をクリックします。")
                    if pipeline_stop_event.is_set():
                        return
                    pyautogui.click(input_x, input_y)
                    if interruptible_sleep(0.25):
                        return
                    if pipeline_stop_event.is_set():
                        return
                    pyautogui.click(input_x, input_y)
                    if interruptible_sleep(0.25):
                        return

                    say(f"[Auto] 入力欄にコード '{code}' を直接打ち込みます！")
                    if pipeline_stop_event.is_set():
                        return
                    mac_safe_type(code)
                    if interruptible_sleep(0.25):
                        return

                    if pipeline_stop_event.is_set():
                        return
                    pyautogui.press('enter')
                    if interruptible_sleep(0.25):
                        return
                    say("[Auto] ★Enterキーでフォーム送信を試行！")

                    if button_bbox:
                        b_xs = [pt[0] for pt in button_bbox]
                        b_ys = [pt[1] for pt in button_bbox]
                        btn_center_x = (min(b_xs) + max(b_xs)) / 2.0
                        btn_center_y = (min(b_ys) + max(b_ys)) / 2.0 + crop_y0
                        btn_x = round(screen_w * (btn_center_x / float(w)))
                        btn_y = round(screen_h * (btn_center_y / float(h)))
                        say(f"[Auto] ★提出ボタン重心=({btn_x}, {btn_y})をクリックします。")
                        if pipeline_stop_event.is_set():
                            return
                        pyautogui.click(btn_x, btn_y)
                    else:
                        fallback_btn_y = round(screen_h * ((ly + 35) / float(h)))
                        say(f"[Auto] 予備座標=({input_x}, {fallback_btn_y}) をクリックします。")
                        if pipeline_stop_event.is_set():
                            return
                        pyautogui.click(input_x, fallback_btn_y)

                    if interruptible_sleep(0.25):
                        return
                    say("★★★【完全自動提出成功】すべての処理が正常終了しました！★★★")
                    if ui_app:
                        ui_app.update_status(f"提出完了: {code}", "success")
                    submitted = True
                    break

                if interruptible_sleep(0.5):
                    return

            if not submitted and not pipeline_stop_event.is_set():
                say("[Auto] 出席入力欄が確認できませんでした。手動で確認してください。")

    except Exception as e:
        say(f"[Auto エラー] {e}")


def alert(panel: Panel, kind: str, prev: Optional[str], frame, elapsed: Optional[float], reader=None):
    if pipeline_stop_event.is_set():
        return
    code, remaining = panel.code, panel.remaining
    try:
        import pyperclip
        pyperclip.copy(code)
    except Exception as e:
        say(f"クリップボードへのコピーに失敗: {e}")

    if kind == "fix":
        title, body = "【訂正】出席コード", f"正しいコード: {code}\n(先ほどの {prev} は読み違いでした)"
    else:
        title, body = "出席コード検知", f"コード: {code}\n{remaining}"

    threading.Thread(target=_beep, daemon=True).start()
    threading.Thread(target=_toast, args=(title, body), daemon=True).start()
    if SAVE_EVIDENCE and frame is not None:
        threading.Thread(target=_save_evidence, args=(frame, code), daemon=True).start()

    if ui_app:
        ui_app.update_status(f"検知: {code} ({remaining})", "working")

    if reader and not pipeline_stop_event.is_set():
        threading.Thread(target=auto_submit_pipeline, args=(code, reader), daemon=True).start()

    speed = f"(赤枠の出現から {elapsed:.1f} 秒)" if elapsed is not None else ""
    if kind == "fix":
        say(f"【訂正】正しいコードは {code} です。コピーを更新しました。")
    else:
        say(f"★ 出席コード検知: {code} / {remaining} {speed} (コピー完了)")


# ==========================================================
# 監視スレッド
# ==========================================================
class MonitorThread(threading.Thread):
    def __init__(self, reader):
        super().__init__(daemon=True)
        self.reader = reader
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        import mss
        factory = getattr(mss, "MSS", None) or mss.mss
        tracker, fails = Tracker(), FailCache()
        say("監視を開始しました (MacBook環境 - システムリセット完了)")

        with factory() as sct:
            mon = sct.monitors[1]
            while self.running and not pipeline_stop_event.is_set():
                t0 = time.time()
                try:
                    frame = np.ascontiguousarray(np.asarray(sct.grab(mon))[:, :, :3])
                    panel = None

                    boxes = find_stop_boxes(frame)
                    if not boxes:
                        tracker.absent(t0)
                    else:
                        tracker.present(t0)
                        if tracker.due(t0):
                            tracker.last_check = t0
                            for box in boxes:
                                if pipeline_stop_event.is_set():
                                    break
                                if fails.known(frame, box, t0):
                                    continue
                                panel, reason, _ = read_panel(self.reader, frame, box)
                                if panel:
                                    break
                                fails.add(frame, box, time.time())

                    if panel and panel.code and not pipeline_stop_event.is_set():
                        now = time.time()
                        kind, prev = tracker.on_panel(panel.code, now)
                        if kind:
                            since = (now - tracker.first_seen) if tracker.first_seen else None
                            alert(panel, kind, prev, frame, since, reader=self.reader)

                except Exception as e:
                    say(f"[監視エラー] {e}")
                    time.sleep(1.0)

                if interruptible_sleep(max(0.0, CHECK_INTERVAL - (time.time() - t0))):
                    break


# ==========================================================
# スレッドセーフ対応 GUI画面
# ==========================================================
class SimpleAttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Taylor's 出席オートメーション (Mac)")
        self.geometry("450x300")
        self.resizable(False, False)
        self.configure(bg="#F5F6F8")

        self.lbl_title = tk.Label(self, text="出席コード自動監視・提出", font=("Helvetica", 14, "bold"), bg="#F5F6F8")
        self.lbl_title.pack(pady=10)

        self.lbl_status = tk.Label(self, text="AIエンジンを初期化中...", font=("Helvetica", 12), fg="#555", bg="#F5F6F8")
        self.lbl_status.pack(pady=5)

        self.txt_log = tk.Text(self, height=8, width=52, font=("Courier", 10), state="disabled", bg="#FFFFFF")
        self.txt_log.pack(pady=10)

        self.btn_toggle = tk.Button(self, text="監視を一時停止", font=("Helvetica", 11), command=self.toggle_monitor, state="disabled")
        self.btn_toggle.pack(pady=5)

        self.monitor_thread = None
        self.reader = None

        threading.Thread(target=self.init_ai_and_start, daemon=True).start()

    def log_message(self, msg: str):
        self.after(0, self._safe_log_message, msg)

    def _safe_log_message(self, msg: str):
        self.txt_log.config(state="normal")
        self.txt_log.insert(tk.END, f"{datetime.datetime.now():%H:%M:%S} {msg}\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state="disabled")

    def update_status(self, text: str, state_type: str = "normal"):
        self.after(0, self._safe_update_status, text, state_type)

    def _safe_update_status(self, text: str, state_type: str = "normal"):
        colors = {"normal": "#2E7D32", "working": "#E65100", "success": "#1565C0"}
        self.lbl_status.config(text=text, fg=colors.get(state_type, "#000"))

    def init_ai_and_start(self):
        import easyocr
        enable_dpi_awareness()
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        pipeline_stop_event.clear()
        self.update_status("🟢 自動監視中（講義画面を開いて待機してください）", "normal")
        self.after(0, lambda: self.btn_toggle.config(state="normal"))

        self.monitor_thread = MonitorThread(self.reader)
        self.monitor_thread.start()

    def toggle_monitor(self):
        if self.monitor_thread and self.monitor_thread.running and not pipeline_stop_event.is_set():
            # システム全体を即座に強制中断
            pipeline_stop_event.set()
            self.monitor_thread.stop()
            self.update_status("⏸ システム停止中（提出・監視中断）", "working")
            self.btn_toggle.config(text="監視を再開")
            say("[システム] 監視およびすべての自動提出処理を即座に停止しました。")
        else:
            # システム全体をリセットして再スタート
            pipeline_stop_event.clear()
            self.monitor_thread = MonitorThread(self.reader)
            self.monitor_thread.start()
            self.update_status("🟢 自動監視中（システムリセット完了）", "normal")
            self.btn_toggle.config(text="監視を一時停止")
            say("[システム] 内部状態をリセットし、出席コード監視を再開しました。")

# ==========================================================
# 【1人1端末厳格制限】ライセンス認証セキュリティモジュール (Mac 完全最適化版)
# ==========================================================
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import messagebox

# ★テスト時はこのまま、本番公開時はネット上のURL（Renderなど）に書き換えます
# ここを書き換える（末尾にスラッシュは付けないでください）
LICENSE_SERVER_URL = "https://attendance-license-server-hwit.onrender.com"
LOCAL_LICENSE_PATH = Path.home() / ".taylors_att_license.dat"

def _get_hardware_fingerprint() -> str:
    """Mac固有のハードウェアUUIDからSHA-256指紋を生成"""
    raw = ""
    try:
        raw = subprocess.check_output(
            "ioreg -rd1 -c IOPlatformExpertDevice | grep IOPlatformUUID",
            shell=True, stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        pass
            
    if not raw or not str(raw).strip():
        raw = platform.node() + platform.machine() + platform.processor()
        
    return hashlib.sha256(str(raw).strip().encode("utf-8")).hexdigest()[:32]


def require_license_check() -> bool:
    """起動時にライセンスを検証。無効ならダイアログを出して認証させる"""
    fp = _get_hardware_fingerprint()

    # 1. すでにMac内に保存されたキーがあるか確認
    if LOCAL_LICENSE_PATH.exists():
        try:
            with open(LOCAL_LICENSE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            key = saved.get("key")
            req_data = json.dumps({"license_key": key, "device_fingerprint": fp}).encode("utf-8")
            req = urllib.request.Request(
                f"{LICENSE_SERVER_URL}/api/v1/verify",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as res:
                if res.status == 200:
                    return True  # 認証済み！そのままアプリを起動
        except Exception:
            pass  # サーバー確認に失敗、または無効化された場合は入力画面へ

    # 2. 未登録または再認証が必要な場合、入力画面を表示
    auth_success = False

    def on_submit():
        nonlocal auth_success
        inp_key = ent.get().strip().upper()
        if not inp_key:
            messagebox.showerror("エラー", "ライセンスキーを入力してください。")
            return
        
        lbl_info.config(text="認証中... しばらくお待ちください", fg="#1565C0")
        dlg.update()

        try:
            req_data = json.dumps({
                "license_key": inp_key,
                "device_fingerprint": fp,
                "device_name": platform.node()
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{LICENSE_SERVER_URL}/api/v1/activate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as res:
                body = json.loads(res.read().decode("utf-8"))
                if body.get("success"):
                    with open(LOCAL_LICENSE_PATH, "w", encoding="utf-8") as f:
                        json.dump({"key": inp_key, "fp": fp}, f)
                    messagebox.showinfo("成功", f"端末の登録が完了しました！\n所有者: {body.get('owner')} 様\n※このMacでのみご利用いただけます。")
                    auth_success = True
                    dlg.destroy()
        except urllib.error.HTTPError as e:
            try:
                err_detail = json.loads(e.read().decode("utf-8")).get("detail", "認証に失敗しました。")
            except Exception:
                err_detail = f"認証エラー ({e.code})"
            lbl_info.config(text=err_detail, fg="#D32F2F")
        except Exception as e:
            lbl_info.config(text="サーバーに接続できません。通信環境を確認してください。", fg="#D32F2F")

    dlg = tk.Tk()
    dlg.title("Taylor's 出席ツール - ライセンス認証 (Mac)")
    dlg.geometry("440x230")
    dlg.resizable(False, False)
    dlg.configure(bg="#F5F6F8")

    tk.Label(dlg, text="【1人1端末限定】ライセンス認証", font=("Helvetica", 14, "bold"), bg="#F5F6F8").pack(pady=12)
    tk.Label(dlg, text="購入時に受け取ったライセンスキーを入力してください:\n(第三者への譲渡・共有は自動検知され停止します)", font=("Helvetica", 10), bg="#F5F6F8", fg="#555").pack()

    ent = tk.Entry(dlg, width=28, font=("Courier", 13, "bold"), justify="center")
    ent.pack(pady=10)

    lbl_info = tk.Label(dlg, text="", font=("Helvetica", 10, "bold"), bg="#F5F6F8")
    lbl_info.pack(pady=2)

    btn = tk.Button(dlg, text="このMacを登録して利用開始", command=on_submit, bg="#1565C0", fg="white", font=("Helvetica", 11, "bold"), padx=10, pady=5)
    btn.pack(pady=8)

    dlg.mainloop()
    return auth_success

def main():
    global ui_app
    
    # ─── ライセンスガード（認証に成功した人のみ起動） ───
    if not require_license_check():
        print("[システム] ライセンス認証が行われなかったため終了します。")
        sys.exit(0)
    # ────────────────────────────────────────────────────

    ui_app = SimpleAttendanceApp()
    ui_app.mainloop()

if __name__ == "__main__":
    main()