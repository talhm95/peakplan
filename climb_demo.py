"""Climbing-analysis DEMO: pose -> COM / base of support / metrics -> annotated MP4s + Hebrew RTL HTML.
Usage: python climb_demo.py out_dir attempt1.MOV attempt2.MOV attempt3.MOV
"""
import sys, os, json, base64, subprocess, io
import numpy as np, cv2
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 15
CLIP_SEC = 8
OUT = sys.argv[1]
INPUTS = sys.argv[2:5]
os.makedirs(OUT, exist_ok=True)

# ---------------- de Leva / Dempster segment model ----------------
# (mass fraction, proximal landmark(s), distal landmark(s), COM fraction from proximal)
def P(lm, idx):
    if isinstance(idx, (list, tuple)):
        return lm[list(idx)].mean(axis=0)
    return lm[idx]

SEGMENTS = [
    (0.070, (7, 8), (7, 8), 0.0),          # head (ear midpoint)
    (0.430, (11, 12), (23, 24), 0.5),      # trunk
    (0.027, 11, 13, 0.436), (0.027, 12, 14, 0.436),   # upper arms
    (0.016, 13, 15, 0.430), (0.016, 14, 16, 0.430),   # forearms
    (0.006, 15, (17, 19), 0.5), (0.006, 16, (18, 20), 0.5),  # hands
    (0.142, 23, 25, 0.433), (0.142, 24, 26, 0.433),   # thighs
    (0.043, 25, 27, 0.433), (0.043, 26, 28, 0.433),   # shanks
    (0.014, 29, 31, 0.5), (0.014, 30, 32, 0.5),       # feet (heel->toe)
]
TOTAL_M = sum(s[0] for s in SEGMENTS)

def com_of(lm):
    c = np.zeros(2)
    for m, a, b, f in SEGMENTS:
        pa, pb = P(lm, a), P(lm, b)
        c += m * (pa + f * (pb - pa))
    return c / TOTAL_M

def movavg(x, w):
    if w <= 1: return x
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(w) / w
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)

def runs(mask):
    out, s = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and s is None: s = i
        if not v and s is not None: out.append((s, i)); s = None
    return out

def contact_mask(pos, tl, thr=0.6, minlen=5):
    v = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * FPS / tl
    still = v < thr
    m = np.zeros(len(pos), bool)
    for s, e in runs(still):
        if e - s >= minlen: m[s:e] = True
    return m

# ---------------- video helpers ----------------
def downscale(src, dst):
    vf = f"scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)',fps={FPS}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", dst], check=True)

def read_frames(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr

def run_pose(frames):
    H, W = frames[0].shape[:2]
    lms, vis, wl = [], [], []
    # low thresholds keep tracking the climber on a busy wall (default 0.5 lost him in ~80% of frames)
    with mp.solutions.pose.Pose(model_complexity=1, static_image_mode=False,
                                min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        for f in frames:
            r = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if r.pose_landmarks:
                a = np.array([[p.x * W, p.y * H, p.visibility] for p in r.pose_landmarks.landmark])
                lms.append(a[:, :2]); vis.append(a[:, 2])
                wl.append(np.array([[p.x, p.y, p.z] for p in r.pose_world_landmarks.landmark]))
            else:
                lms.append(np.full((33, 2), np.nan)); vis.append(np.zeros(33)); wl.append(np.full((33, 3), np.nan))
    lms, vis, wl = np.array(lms), np.array(vis), np.array(wl)
    # drop unreliable leg/foot points (low visibility) and let them be filled from neighbouring good frames
    bad = vis[:, 25:33] < 0.5
    lms[:, 25:33][bad] = np.nan; wl[:, 25:33][bad] = np.nan
    # interpolate gaps
    t = np.arange(len(lms))
    for arr in (lms, wl):
        for j in range(33):
            for d in range(arr.shape[2]):
                y = arr[:, j, d]; ok = ~np.isnan(y)
                if ok.sum() > 1: arr[:, j, d] = np.interp(t, t[ok], y[ok])
    run_pose.wl = wl  # metric world landmarks (meters, hip-centred) of the last call
    return lms, vis, W, H

def pick_window(lms, vis, W, H):
    n = len(lms); win = CLIP_SEC * FPS
    if n <= win: return 0, n
    key = [11, 12, 23, 24, 27, 28, 29, 30, 31, 32]
    inframe = ((lms[:, key, 0] > 0) & (lms[:, key, 0] < W) & (lms[:, key, 1] > 0) & (lms[:, key, 1] < H)).all(1)
    full = inframe & (vis[:, key].min(1) > 0.5)
    lf = lms[:, [29, 31]].mean(1); rf = lms[:, [30, 32]].mean(1)
    fm = np.linalg.norm(np.gradient(lf, axis=0), axis=1) + np.linalg.norm(np.gradient(rf, axis=0), axis=1)
    fm = np.clip(fm, 0, np.percentile(fm, 98))  # ignore tracking jumps
    best, bs = -1, 0
    for s in range(0, n - win, 3):
        vf = full[s:s + win].mean()
        score = fm[s:s + win].sum() * (vf if vf > 0.8 else vf * 0.2)
        if score > best: best, bs = score, s
    return bs, bs + win

# ---------------- analysis ----------------
def analyse(lms):
    n = len(lms)
    sm = movavg(lms.reshape(n, -1), 3).reshape(n, 33, 2)
    mid_sh = sm[:, [11, 12]].mean(1); mid_hip = sm[:, [23, 24]].mean(1)
    tl = float(np.median(np.linalg.norm(mid_sh - mid_hip, axis=1)))
    com = movavg(np.array([com_of(f) for f in sm]), 5)
    feet = [sm[:, [29, 31]].mean(1), sm[:, [30, 32]].mean(1)]
    hands = [sm[:, [15, 17, 19]].mean(1), sm[:, [16, 18, 20]].mean(1)]
    fc = [contact_mask(f, tl) for f in feet]
    hc = [contact_mask(h, tl) for h in hands]
    inside = np.full(n, np.nan); dev = np.full(n, np.nan)
    des_com = np.full((n, 2), np.nan); des_feet = [np.full((n, 2), np.nan) for _ in feet]
    hulls = []
    for i in range(n):
        pts = [feet[k][i] for k in range(2) if fc[k][i]] + [hands[k][i] for k in range(2) if hc[k][i]]
        hull = None
        if pts:
            pts = np.array(pts, np.float32)
            cen = pts.mean(0)
            des_com[i] = (cen[0], com[i, 1])
            dev[i] = abs(com[i, 0] - cen[0]) / tl
            if len(pts) >= 3:
                hull = cv2.convexHull(pts)
                inside[i] = cv2.pointPolygonTest(hull, tuple(map(float, com[i])), False) >= 0
            else:
                lo, hi = pts[:, 0].min() - 0.15 * tl, pts[:, 0].max() + 0.15 * tl
                inside[i] = lo <= com[i, 0] <= hi
                hull = pts
        hulls.append(hull)
        pf = [k for k in range(2) if fc[k][i]]
        if pf:
            mid = np.mean([feet[k][i] for k in pf], 0)
            dx = com[i, 0] - mid[0]
            for k in pf:
                des_feet[k][i] = feet[k][i] + (dx, 0)
    # metrics
    c = com / tl
    vel = np.gradient(c, axis=0) * FPS
    speed = np.linalg.norm(vel, axis=1)
    L = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
    T = n / FPS
    cj = movavg(c, 7)
    jerk = np.gradient(np.gradient(np.gradient(cj, axis=0), axis=0), axis=0) * FPS ** 3
    nj = float(np.sqrt(0.5 * np.sum(np.sum(jerk ** 2, 1)) / FPS * T ** 5 / max(L, 1e-6) ** 2))
    hp = cv2.convexHull(c.astype(np.float32))
    per = float(cv2.arcLength(hp, True))
    gie = float(np.log2(2 * L / per)) if per > 0 else 0.0
    readj = sum(1 for m in fc for s, e in runs(m) if (e - s) / FPS < 0.5)
    metrics = dict(
        dev=float(np.nanmean(dev)),
        out=float(100 * np.nanmean(1 - inside)) if np.isfinite(inside).any() else 0.0,
        readj=int(readj),
        jerk=float(np.log10(nj)),
        gie=gie,
        immob=float(100 * np.mean(speed < 0.15)),
    )
    return dict(sm=sm, com=com, tl=tl, feet=feet, fc=fc, des_com=des_com, des_feet=des_feet,
                inside=inside, hulls=hulls, dev=dev), metrics

# ---------------- rendering ----------------
CONN = list(mp.solutions.pose.POSE_CONNECTIONS)

def render(frames, A, dst):
    H, W = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                          "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    ip = lambda v: (int(v[0]), int(v[1]))
    for i, f in enumerate(frames):
        img = f.copy()
        lm = A["sm"][i]
        hull = A["hulls"][i]
        if hull is not None and len(hull) >= 3:
            ov = img.copy(); cv2.fillPoly(ov, [hull.reshape(-1, 2).astype(np.int32)], (255, 200, 80))
            img = cv2.addWeighted(ov, 0.18, img, 0.82, 0)
        for a, b in CONN:
            if a > 10 and b > 10:
                cv2.line(img, ip(lm[a]), ip(lm[b]), (235, 235, 235), 1, cv2.LINE_AA)
        # trail
        for j in range(max(0, i - 30), i):
            al = (j - i + 30) / 30
            cv2.line(img, ip(A["com"][j]), ip(A["com"][j + 1]), (0, 0, int(120 + 135 * al)), max(1, int(3 * al)), cv2.LINE_AA)
        c = A["com"][i]; d = A["des_com"][i]
        if np.isfinite(d).all():
            cv2.line(img, ip(c), ip(d), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, ip(d), 11, (80, 220, 80), 2, cv2.LINE_AA)
        for k in range(2):
            fp = A["feet"][k][i]
            col = (0, 140, 255) if A["fc"][k][i] else (0, 90, 160)
            cv2.circle(img, ip(fp), 6, col, -1, cv2.LINE_AA)
            df = A["des_feet"][k][i]
            if np.isfinite(df).all() and abs(df[0] - fp[0]) > 0.1 * A["tl"]:
                cv2.arrowedLine(img, ip(fp), ip(df), (80, 220, 80), 2, cv2.LINE_AA, tipLength=0.25)
                cv2.circle(img, ip(df), 9, (80, 220, 80), 1, cv2.LINE_AA)
                cv2.circle(img, ip(df), 3, (80, 220, 80), -1, cv2.LINE_AA)
        cv2.circle(img, ip(c), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, ip(c), 7, (255, 255, 255), 1, cv2.LINE_AA)
        # gauge
        ins = A["inside"][i]
        txt, col = ("COM IN SUPPORT", (80, 200, 80)) if ins == 1 else (("COM OUT", (60, 60, 230)) if ins == 0 else ("NO CONTACT", (150, 150, 150)))
        cv2.rectangle(img, (10, 10), (215, 62), (30, 30, 30), -1)
        cv2.circle(img, (30, 30), 9, col, -1, cv2.LINE_AA)
        cv2.putText(img, txt, (46, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        dv = A["dev"][i]; dv = 0 if not np.isfinite(dv) else min(dv, 1.0)
        cv2.rectangle(img, (20, 46), (205, 54), (80, 80, 80), -1)
        cv2.rectangle(img, (20, 46), (20 + int(185 * dv), 54), col, -1)
        p.stdin.write(img.tobytes())
    p.stdin.close(); p.wait()

# ---------------- extra parameters (landing, momentum, wall distance) ----------------
def b64img(img, q=82):
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()

def crop_resize(img, c, w, h, out=(480, 360)):
    H, W = img.shape[:2]
    w, h = int(min(w, W)), int(min(h, H))
    x0 = int(np.clip(c[0] - w / 2, 0, W - w)); y0 = int(np.clip(c[1] - h / 2, 0, H - h))
    return cv2.resize(img[y0:y0 + h, x0:x0 + w], out, interpolation=cv2.INTER_CUBIC)

def dashed_ellipse(img, c, ax, col, th=2):
    for a in range(0, 360, 24):
        cv2.ellipse(img, (int(c[0]), int(c[1])), ax, 0, a, a + 13, col, th, cv2.LINE_AA)

TEAL, YEL, CYAN, ORANGE = (122, 134, 18), (40, 190, 245), (220, 220, 0), (40, 150, 245)

def foot_img(frame, heel, toe, foot, des, ang, tl):
    img = frame.copy(); ip = lambda v: (int(v[0]), int(v[1]))
    r = max(6, int(0.18 * tl))
    dashed_ellipse(img, des, (int(r * 1.6), r), TEAL, 2)
    L = int(0.9 * tl)
    for x in range(int(heel[0]), int(heel[0]) + L, 6):
        cv2.circle(img, (x, int(heel[1])), 1, (255, 255, 255), -1)
    v = (toe - heel) / (np.linalg.norm(toe - heel) + 1e-6)
    if v[0] < 0: v = -v
    cv2.line(img, ip(heel), ip(heel + v * L), YEL, 2, cv2.LINE_AA)
    cv2.circle(img, ip(foot), max(4, r // 3), TEAL, -1, cv2.LINE_AA)
    cv2.circle(img, ip(foot), max(4, r // 3), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{ang:.0f}", (int(heel[0] + L * 0.7), int(heel[1] - 0.2 * tl)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YEL, 2, cv2.LINE_AA)
    return b64img(crop_resize(img, (foot + des) / 2, 2.4 * tl, 1.8 * tl))

def extras(A, wl, frames, cmpp):
    sm, tl, n = A["sm"], A["tl"], len(A["sm"])
    E = {}
    # --- landing accuracy: the foot landing that follows the biggest foot travel
    best = None
    for k in range(2):
        for s, e in runs(A["fc"][k]):
            if s < 2: continue
            pre = A["feet"][k][max(0, s - 10):s + 1]
            tr = float(np.linalg.norm(pre[-1] - pre[0]))
            if best is None or tr > best[0]: best = (tr, k, s)
    if best:
        _, k, i = best
        foot = A["feet"][k][i]; des = A["des_feet"][k][i]
        if not np.isfinite(des).all(): des = foot
        heel, toe = sm[i, 29 + k], sm[i, 31 + k]
        # foot pitch from metric 3D world landmarks (camera behind climber makes the 2D angle meaningless)
        wv = wl[max(0, i - 1):i + 2, 31 + k].mean(0) - wl[max(0, i - 1):i + 2, 29 + k].mean(0)
        ang = abs(float(np.degrees(np.arctan2(-wv[1], np.hypot(wv[0], wv[2])))))
        E.update(land_pos=float(np.linalg.norm(des - foot) * cmpp), land_ang=float(ang), land_t=round(i / FPS, 1),
                 land_foot="שמאל" if k == 0 else "ימין", land_img=foot_img(frames[i], heel, toe, foot, des, ang, tl))
    else:
        E.update(land_pos=None, land_ang=None, land_t=None, land_foot=None, land_img=None)
    # --- momentum: hip speed / direction
    hip = movavg(sm[:, [23, 24]].mean(1), 5)
    vel = np.gradient(hip, axis=0) * FPS
    sp = movavg((np.linalg.norm(vel, axis=1) * cmpp / 100)[:, None], 7)[:, 0]  # m/s, smoothed
    lo, hi = 6, max(7, n - 6)
    i = lo + int(np.argmax(sp[lo:hi]))
    E.update(speed=float(sp[i]), dir=float(np.degrees(np.arctan2(-vel[i, 1], vel[i, 0]))), mom_t=round(i / FPS, 1),
             speed_curve=[round(float(x), 3) for x in np.interp(np.linspace(0, n - 1, 40), np.arange(n), sp)])
    E["_mom"] = (frames[i], hip[i], tl)
    # --- distance hip->wall plane (plane = depth of planted hands/feet, MediaPipe world z)
    hz = wl[:, [23, 24], 2].mean(1)
    d = np.full(n, np.nan)
    for j in range(n):
        idx = [[29, 31], [30, 32]]
        zs = [wl[j, idx[k], 2].mean() for k in range(2) if A["fc"][k][j]]
        if zs: d[j] = (np.mean(zs) - hz[j]) * 100
    d = np.abs(d)
    if np.isfinite(d).any():
        t = np.arange(n); ok = np.isfinite(d); d = np.interp(t, t[ok], d[ok])
        E.update(wall=float(np.median(d)), wall_curve=[round(float(x), 1) for x in np.interp(np.linspace(0, n - 1, 40), t, d)])
    else:
        E.update(wall=None, wall_curve=[])
    mid = n // 2
    img = frames[mid].copy()
    cv2.circle(img, (int(A["com"][mid][0]), int(A["com"][mid][1])), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(A["com"][mid][0]), int(A["com"][mid][1])), 8, (255, 255, 255), 2, cv2.LINE_AA)
    E["wall_img"] = b64img(img)
    return E

def mom_img(frame, hip, tl, d, dref):
    img = frame.copy(); L = 1.6 * tl
    def arrow(deg, col, dashed=False):
        e = hip + L * np.array([np.cos(np.radians(deg)), -np.sin(np.radians(deg))])
        if dashed:
            for a in np.linspace(0, 0.85, 7):
                p0 = hip + (e - hip) * a; p1 = hip + (e - hip) * (a + 0.07)
                cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, 3, cv2.LINE_AA)
            cv2.arrowedLine(img, tuple(map(int, hip + (e - hip) * 0.9)), tuple(map(int, e)), col, 3, cv2.LINE_AA, tipLength=0.9)
        else:
            cv2.arrowedLine(img, tuple(map(int, hip)), tuple(map(int, e)), col, 3, cv2.LINE_AA, tipLength=0.12)
    if dref is not None: arrow(dref, ORANGE, True)
    arrow(d, CYAN)
    cv2.circle(img, tuple(map(int, hip)), 7, TEAL, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(map(int, hip)), 7, (255, 255, 255), 2, cv2.LINE_AA)
    return b64img(img)

def sc(x):
    return int(round(max(0.0, min(100.0, x))))

PARAMS = [  # key, Hebrew label, unit
    ("pos", "סטיית מיקום", "ס״מ"), ("ang", "סטיית זווית", "°"), ("speed", "מהירות", "m/s"),
    ("dir", "כיוון", "°"), ("wall", "מרחק מהקיר", "ס״מ"),
]

def finalize(results):
    """Pick the reference segment and score every parameter 0-100."""
    def target_score(r):
        p = 0 if r["land_pos"] is None else 100 * (1 - r["land_pos"] / 15)
        a = 0 if r["land_ang"] is None else 100 * (1 - r["land_ang"] / 45)
        c = 100 * (1 - r["dev"] / 0.5)
        return sc(p) + sc(a) + sc(c)
    ref = int(np.argmax([target_score(r) for r in results]))
    R = results[ref]
    for r in results:
        S = {
            "pos": None if r["land_pos"] is None else sc(100 * (1 - r["land_pos"] / 15)),
            "ang": None if r["land_ang"] is None else sc(100 * (1 - r["land_ang"] / 45)),
            "speed": sc(100 * (1 - abs(r["speed"] - R["speed"]) / max(R["speed"], 1e-3))),
            "dir": sc(100 * (1 - abs(((r["dir"] - R["dir"] + 180) % 360) - 180) / 60)),
            "wall": None if (r["wall"] is None or R["wall"] is None) else sc(100 * (1 - abs(r["wall"] - R["wall"]) / max(R["wall"], 10))),
        }
        vals = [v for v in S.values() if v is not None]
        r["scores"] = S
        r["overall"] = round(float(np.mean(vals)), 1) if vals else 0.0
        r["land_score"] = sc(np.mean([v for v in (S["pos"], S["ang"]) if v is not None])) if (S["pos"] is not None) else None
        r["mom_score"] = sc((S["speed"] + S["dir"]) / 2)
        fr, hip, tl = r.pop("_mom")
        r["mom_img"] = mom_img(fr, hip, tl, r["dir"], None if r is R else R["dir"])
        r["dir_diff"] = round(abs(((r["dir"] - R["dir"] + 180) % 360) - 180))
    return ref

# ---------------- HTML ----------------
TREND = [  # key, Hebrew label, unit
    ("dev", "סטיית מרכז מסה", "אורכי גו"), ("out", "זמן מחוץ לבסיס התמיכה", "%"),
    ("readj", "תיקוני רגל", "פעמים"), ("jerk", "Jerk מנורמל", "log₁₀"),
    ("gie", "אנטרופיה גאומטרית", "GIE"), ("immob", "יחס חוסר תנועה", "%"),
]

def chart_png(vals):
    fig, ax = plt.subplots(figsize=(3.2, 1.4), dpi=110)
    x = list(range(1, len(vals) + 1))
    ax.plot(x, vals, color="#12867A", lw=2.2, marker="o", ms=6, mfc="white", mew=2)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.2f}".rstrip("0").rstrip("."), (xi, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([f"#{i}" for i in x], fontsize=8)
    ax.tick_params(axis="y", labelsize=7, colors="#888")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#ddd")
    ax.margins(x=0.15, y=0.35); ax.grid(axis="y", color="#f0f0f0")
    fig.tight_layout(pad=0.3)
    b = io.BytesIO(); fig.savefig(b, format="png", transparent=True); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

HTML = r"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ניתוח טיפוס – DEMO</title>
<style>
:root{--bg:#FAF6F0;--card:#fff;--ink:#14213D;--mut:#6b7280;--acc:#12867A;--acc2:#0E6B62;--nav:#14284B;--g:#2E9E4F;--y:#F2B233;--r:#E5484D;--line:#e8e2d8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font-family:system-ui,-apple-system,"Segoe UI",Arial,sans-serif;color:var(--ink)}
.app{display:flex;min-height:100vh}
nav{width:200px;background:var(--nav);color:#fff;padding:22px 12px;flex-shrink:0;position:sticky;top:0;height:100vh}
.logo{font-weight:800;letter-spacing:.5px;text-align:center;font-size:17px;padding-bottom:18px;border-bottom:1px solid #ffffff26;margin-bottom:14px}
.logo small{display:block;font-weight:500;font-size:11px;opacity:.6;letter-spacing:0}
.nb{display:flex;align-items:center;gap:10px;width:100%;border:0;background:none;color:#dfe6f3;font-size:15px;padding:12px 14px;border-radius:12px;cursor:pointer;margin-bottom:4px;text-align:right;font-family:inherit}
.nb svg{width:20px;height:20px;flex-shrink:0}.nb.on{background:var(--acc);color:#fff;font-weight:700}.nb:hover:not(.on){background:#ffffff14}
main{flex:1;padding:22px 28px;min-width:0}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
h1{margin:0;font-size:30px}.sub{color:var(--mut);margin-top:4px;font-size:15px}
.btns{display:flex;gap:8px;flex-wrap:wrap}.btn{border:1.5px solid var(--ink);background:#fff;border-radius:12px;padding:9px 16px;font-size:14px;font-weight:600;cursor:pointer;font-family:inherit;color:var(--ink)}
.btn.p{background:var(--acc);border-color:var(--acc);color:#fff}
.seg{display:flex;gap:6px;margin:16px 0;flex-wrap:wrap}.sg{border:1px solid var(--line);background:#fff;border-radius:12px;padding:8px 16px;font-size:14px;font-weight:600;cursor:pointer;font-family:inherit;color:var(--ink)}
.sg.on{background:var(--ink);color:#fff;border-color:var(--ink)}.sg .rf{font-size:11px;background:#d9f2ec;color:var(--acc2);border-radius:8px;padding:1px 6px;margin-right:6px}
.sg.on .rf{background:#ffffff30;color:#fff}
.grid{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr) minmax(0,.8fr);gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 16px}
.media{padding:6px;overflow:hidden}.media img,.media video{width:100%;border-radius:14px;display:block;background:#000}
.k{font-size:15px;font-weight:700}.big{font-size:38px;font-weight:800;line-height:1.1;margin-top:4px}.big small{font-size:16px;color:var(--mut);font-weight:500}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px}
.pill{display:inline-block;border-radius:10px;padding:4px 10px;font-weight:800;color:#fff;font-size:15px;white-space:nowrap}.pill small{font-weight:500;opacity:.85;font-size:12px}
.g{background:var(--g)}.y{background:var(--y)}.r{background:var(--r)}.n{background:#9ca3af}
.mut{color:var(--mut);font-size:12.5px}.stack{display:flex;flex-direction:column;gap:12px}
.note{background:#e6f4f1;border:1px solid #cde9e3;border-radius:16px;padding:12px 16px;margin-top:14px;font-size:14px}
.warn{background:#fff7e0;border-color:#f3e2b3;color:#6b5a2a;font-size:12.5px}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:16px;overflow:hidden;border:1px solid var(--line)}
th{background:#f3efe8;font-size:14px;padding:10px;font-weight:700}td{text-align:center;padding:8px;border-top:1px solid var(--line);font-size:14px}
tr.ref td{background:#eaf7f3}td .pill{font-size:13px;padding:2px 9px;margin-top:3px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:12px 0}
.tile{background:#fff;border:1px solid var(--line);border-radius:14px;padding:6px;text-align:center;cursor:pointer}.tile img{width:100%;height:110px;object-fit:cover;border-radius:10px}
.tile.ref{border:2px solid var(--acc);background:#eaf7f3}.tile b{display:block;margin-top:4px}
.legend{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0;font-size:13px}.legend span{border-radius:10px;padding:4px 10px;background:#fff;border:1px solid var(--line)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-left:5px;vertical-align:middle}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.trend{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.trend img{width:100%;direction:ltr}
svg.ch{width:100%;height:150px;direction:ltr}
.lg{font-size:12px;color:var(--mut);display:flex;gap:12px}.lg i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-left:4px;vertical-align:middle}
.page{display:none}.page.on{display:block}
@media(max-width:1000px){.app{flex-direction:column}nav{width:auto;height:auto;position:static;display:flex;overflow-x:auto;gap:4px;padding:10px}.logo{display:none}.nb{width:auto;white-space:nowrap}
.grid,.two,.trend{grid-template-columns:1fr}main{padding:16px}}
</style></head><body><div class="app">
<nav><div class="logo">ניתוח טיפוס<small>DEMO · ניתוח וידאו</small></div>
<button class="nb on" data-p="ov"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 20V12M12 20V5M19 20v-9"/></svg>סקירה</button>
<button class="nb" data-p="land"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg>דיוק נחיתה</button>
<button class="nb" data-p="mom"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17l6-6 4 4 8-8M15 7h6v6"/></svg>מומנטום</button>
<button class="nb" data-p="wall"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 12h16M8 8l-4 4 4 4M16 8l4 4-4 4"/></svg>מרחק מהקיר</button>
<button class="nb" data-p="cmp"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5"/></svg>השוואת ניסיונות</button>
</nav><main>
<div class="top"><div><h1 id="title"></h1><div class="sub" id="sub"></div></div>
<div class="btns"><button class="btn" id="exp">⬇ ייצוא נתונים</button></div></div>
<div class="seg" id="seg"></div>

<section class="page on" id="p-ov">
 <div class="grid"><div class="card media"><video id="vid" controls autoplay muted loop playsinline></video>
 <div class="legend" style="margin:8px 4px 2px"><span><i class="dot" style="background:#ff2a2a"></i>מרכז מסה</span><span><i class="dot" style="border:2px solid #50dc50"></i>מרכז מסה רצוי*</span><span><i class="dot" style="background:#ff8c00"></i>רגליים</span><span><i class="dot" style="background:#50dc50"></i>מיקום רגל רצוי*</span></div></div>
 <div class="stack" id="ov-scores"></div><div class="stack" id="ov-cards"></div></div>
 <div class="card" style="margin-top:14px"><div class="row"><div class="k">מגמה בין הקטעים</div><div class="mut" id="takeaway" style="font-weight:700;color:var(--acc2);font-size:14px"></div></div><div class="trend" id="trend" style="margin-top:8px"></div></div>
</section>

<section class="page" id="p-land">
 <div class="grid"><div class="card media"><img id="land-frame"></div>
 <div class="stack"><div class="card media"><img id="land-img"><div class="legend" style="margin:8px 4px 2px"><span><i class="dot" style="border:2px dashed #12867A"></i>יעד המיקום*</span><span><i class="dot" style="background:#12867A"></i>נקודת מגע</span><span><i class="dot" style="background:#F5BE28"></i>זווית כף הרגל</span></div></div>
 <div class="card"><div class="k">ציר זמן – מהירות כף הרגל</div><svg class="ch" id="land-ch"></svg></div></div>
 <div class="stack" id="land-cards"></div></div>
 <div class="note" id="land-note"></div>
</section>

<section class="page" id="p-mom">
 <div class="grid"><div class="card media"><img id="mom-img"><div class="legend" style="margin:8px 4px 2px"><span><i class="dot" style="background:#00dcdc"></i>כיוון התנועה</span><span><i class="dot" style="background:#F59628"></i>כיוון ייחוס</span></div></div>
 <div class="stack"><div class="card"><div class="row"><div class="k">מהירות האגן לאורך הקטע</div><div class="lg" id="mom-lg"></div></div><svg class="ch" id="mom-ch"></svg></div>
 <div class="card" id="mom-ref"></div></div>
 <div class="stack" id="mom-cards"></div></div>
 <div class="note" id="mom-note"></div>
</section>

<section class="page" id="p-wall">
 <div class="grid"><div class="card media"><img id="wall-img"></div>
 <div class="stack"><div class="card"><div class="k">מבט צד (סכמטי)</div><svg id="wall-side" viewBox="0 0 320 210" style="width:100%;height:210px;direction:ltr"></svg></div>
 <div class="card"><div class="row"><div class="k">מרחק לאורך התנועה</div><div class="lg" id="wall-lg"></div></div><svg class="ch" id="wall-ch"></svg></div></div>
 <div class="stack" id="wall-cards"></div></div>
 <div class="note warn">מרחק מהקיר מוערך מעומק (z) של MediaPipe ממצלמה אחת, ביחס למישור שבין כפות הרגליים הנשענות. זו הערכה גסה בלבד; בגרסה המלאה נדרש כיול/מודל תלת-ממדי.</div>
</section>

<section class="page" id="p-cmp">
 <div class="tiles" id="tiles"></div>
 <table id="tbl"></table>
 <div class="legend"><span><i class="dot g"></i>80–100 קרוב ליעד</span><span><i class="dot y"></i>60–79 סטייה בינונית</span><span><i class="dot r"></i>0–59 סטייה גבוהה</span><span style="border:0;background:none" class="mut"><b>ציון כולל:</b> ממוצע חמשת המדדים במשקל שווה</span></div>
 <div class="two"><div class="card"><div class="k">💪 נקודת חוזק</div><div id="strong" style="margin-top:6px"></div></div>
 <div class="card"><div class="k">↻ דפוסים חוזרים</div><div id="patterns" style="margin-top:6px"></div></div></div>
</section>

<div class="note warn">* ערכי "רצוי"/"יעד" הם היוריסטיקה לצורך הדגמה (מרכז מסה מעל מרכז בסיס התמיכה; יעד רגל = נקודה שמציבה את אמצע כפות הרגליים מתחת למרכז המסה). ציוני מהירות, כיוון ומרחק מהקיר מחושבים ביחס ל<b>קטע הייחוס</b> (הקטע עם דיוק הנחיתה ומרכז המסה הטובים ביותר). בגרסה המלאה היעדים יגיעו ממתכנן המסלול. סנטימטרים מוערכים לפי אורך הגו במודל MediaPipe.</div>
</main></div>
<script>
const D=__DATA__, G=__GLOBAL__;
let cur=G.ref;
const $=id=>document.getElementById(id);
const W=G.word, NM=i=>W+' '+(i+1);
const cls=v=>v==null?'n':v>=80?'g':v>=60?'y':'r';
const pill=v=>`<span class="pill ${cls(v)}">${v==null?'—':v}<small> / 100</small></span>`;
const f=(v,d=1)=>v==null?'—':(+v).toFixed(d);
function line(id,series,ylab){const s=$(id),w=600,h=150,p=26;let all=series.flatMap(x=>x.v).filter(x=>x!=null);if(!all.length){s.innerHTML='';return}
 let mn=Math.min(0,...all),mx=Math.max(...all)*1.15||1;const X=(i,n)=>p+i*(w-2*p)/(n-1),Y=v=>h-p-(v-mn)/(mx-mn)*(h-2*p);
 let o=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"></svg>`;s.setAttribute('viewBox',`0 0 ${w} ${h}`);s.setAttribute('preserveAspectRatio','none');
 let g=`<line x1="${p}" y1="${h-p}" x2="${w-p}" y2="${h-p}" stroke="#e5e0d6"/><text x="${p}" y="12" font-size="11" fill="#999">${ylab}</text>`;
 series.forEach(x=>{const pts=x.v.map((v,i)=>X(i,x.v.length)+','+Y(v)).join(' ');
  g+=`<polygon points="${p},${h-p} ${pts} ${w-p},${h-p}" fill="${x.c}" opacity="${x.fill?0.12:0}"/><polyline points="${pts}" fill="none" stroke="${x.c}" stroke-width="3" ${x.dash?'stroke-dasharray="7 6"':''} stroke-linejoin="round"/>`;
  if(x.mark!=null){const i=x.mark;g+=`<circle cx="${X(i,x.v.length)}" cy="${Y(x.v[i])}" r="6" fill="#fff" stroke="${x.c}" stroke-width="3"/>`}});
 s.innerHTML=g}
function seg(){$('seg').innerHTML=D.map((r,i)=>`<button class="sg ${i==cur?'on':''}" data-i="${i}">${NM(i)}${i==G.ref?'<span class="rf">ייחוס</span>':''}</button>`).join('');
 document.querySelectorAll('.sg').forEach(b=>b.onclick=()=>{cur=+b.dataset.i;render()})}
const TITLES={ov:'סקירה',land:'דיוק נחיתה — מיקום וזווית',mom:'מהירות וכיוון התנועה',wall:'מרחק מרכז המסה מהקיר',cmp:'השוואת ניסיונות'};
let page='ov';
function card(k,big,unit,sub,score){return `<div class="card"><div class="row"><div class="k">${k}</div>${score!==undefined?pill(score):''}</div><div class="big"><bdi dir="ltr">${big}</bdi> <small>${unit||''}</small></div>${sub?`<div class="mut">${sub}</div>`:''}</div>`}
function render(){const r=D[cur],R=D[G.ref];seg();
 $('title').textContent=TITLES[page];$('sub').textContent=G.climber+' · '+G.route+' / '+NM(cur)+' ('+r.start+'–'+r.end+' שנ׳)';
 // overview
 const v=$('vid');if(v.dataset.i!=cur){v.src=r.video;v.dataset.i=cur;v.play().catch(()=>{})}
 $('ov-scores').innerHTML=`<div class="card"><div class="k">ציון כולל</div><div class="big">${f(r.overall)} <small>/ 100</small></div><div class="mut">ממוצע חמשת המדדים</div></div>`+
  card('דיוק נחיתה',r.land_score==null?'—':r.land_score,'/ 100','מיקום + זווית')+card('מומנטום',r.mom_score,'/ 100','מהירות + כיוון')+
  card('מרחק מהקיר',f(r.wall,0),'ס״מ','ייחוס: '+f(R.wall,0)+' ס״מ',r.scores.wall);
 const T=G.trendKeys;$('ov-cards').innerHTML=T.map(([k,l,u])=>`<div class="card" style="padding:10px 14px"><div class="mut">${l}</div><div style="font-size:22px;font-weight:800">${k=='readj'?r[k]:f(r[k],k=='dev'||k=='gie'?2:k=='jerk'?1:0)} <small class="mut">${u}</small></div></div>`).join('');
 $('takeaway').textContent=G.takeaway;$('trend').innerHTML=G.trend.map(t=>`<div><div class="mut" style="font-weight:600">${t.l} (${t.u})</div><img src="${t.img}"></div>`).join('');
 // landing
 $('land-frame').src=r.wall_img;$('land-img').src=r.land_img||'';
 $('land-cards').innerHTML=r.land_pos==null?card('דיוק נחיתה','—','','לא זוהתה נחיתה בקטע זה'):
  card('סטייה במיקום',f(r.land_pos),'ס״מ','יעד: 0 ס״מ · רגל '+r.land_foot,r.scores.pos)+card('סטייה בזווית',f(r.land_ang,0)+'°','','יעד: 0° (כף רגל שטוחה, הטיה תלת-ממדית)',r.scores.ang)+
  `<div class="card"><div class="k">🏆 ציון דיוק הנחיתה</div><div class="big"><span class="pill ${cls(r.land_score)}" style="font-size:30px">${r.land_score} / 100</span></div><div class="mut">ממוצע מיקום וזווית · מגע ב-${r.land_t} שנ׳</div></div>`;
 line('land-ch',[{v:r.speed_curve,c:'#12867A',fill:1,mark:r.land_t==null?null:Math.round(r.land_t/((r.end-r.start)||1)*39)}],'מהירות אגן m/s');
 $('land-note').innerHTML='<b>📋 כיוון כף הרגל במגע הראשון:</b> '+(r.land_ang==null?'—':r.land_ang>20?'כף הרגל נטויה ב-'+f(r.land_ang,0)+'° – לנסות להניח אותה שטוחה יותר על האחיזה.':'זווית כף הרגל קרובה ליעד.');
 // momentum
 $('mom-img').src=r.mom_img;
 $('mom-cards').innerHTML=card('מהירות האגן',f(r.speed,2),'m/s','שיא ב-'+r.mom_t+' שנ׳ · ייחוס '+f(R.speed,2),r.scores.speed)+card('כיוון התנועה',f(r.dir,0)+'°','','יחסית לאופק · הפרש '+r.dir_diff+'° מהייחוס',r.scores.dir)+
  `<div class="card"><div class="k">ציון תנועה</div><div class="big" style="color:var(--${cls(r.mom_score)})">${r.mom_score}<small> / 100</small></div><div class="mut">ממוצע מהירות וכיוון</div></div>`;
 $('mom-ref').innerHTML=`<div class="row mut" style="font-size:14px"><span>${NM(G.ref)} ייחוס</span><span>מהירות יעד: <b>${f(R.speed,2)} m/s</b></span><span>כיוון יעד: <b>${f(R.dir,0)}°</b></span></div>`;
 const ms=[{v:r.speed_curve,c:'#12867A',fill:1}];if(cur!=G.ref)ms.push({v:R.speed_curve,c:'#F2B233',dash:1});line('mom-ch',ms,'m/s');
 $('mom-lg').innerHTML=`<span><i style="background:#12867A"></i>${NM(cur)}</span>`+(cur!=G.ref?`<span><i style="background:#F2B233"></i>ייחוס</span>`:'');
 $('mom-note').innerHTML='<b>📋 הערה:</b> '+(cur==G.ref?'זה קטע הייחוס – שאר הקטעים מושווים אליו.':r.dir_diff>20?'למקד את כיוון התנועה ביחס לקטע הייחוס (הפרש '+r.dir_diff+'°).':'כיוון התנועה קרוב לייחוס.');
 // wall
 $('wall-img').src=r.wall_img;
 $('wall-cards').innerHTML=card('מרחק מהקיר',f(r.wall,0),'ס״מ','התאמה למרחק הייחוס',r.scores.wall)+card('מרחק בייחוס',f(R.wall,0),'ס״מ')+
  card('הפרש מהייחוס',(r.wall!=null&&R.wall!=null?(r.wall-R.wall>=0?'+':'')+f(r.wall-R.wall,0):'—'),'ס״מ')+
  `<div class="card" style="background:#eef7f5"><div class="k">⚡ חיסכון באנרגיה</div><div class="big" style="font-size:26px">לא הוערך</div><div class="mut">נדרש מודל מאומת</div></div>`;
 const sx=d=>250-Math.min(d||0,45)*5;
 $('wall-side').innerHTML=`<rect x="252" y="5" width="22" height="200" fill="#e7e2da"/><line x1="252" y1="5" x2="252" y2="205" stroke="#bbb"/>`+
  (cur!=G.ref?`<g opacity=".55"><circle cx="${sx(R.wall)}" cy="60" r="14" fill="#F2B233"/><line x1="${sx(R.wall)}" y1="74" x2="${sx(R.wall)}" y2="140" stroke="#F2B233" stroke-width="14" stroke-linecap="round"/></g>`:'')+
  `<circle cx="${sx(r.wall)}" cy="60" r="14" fill="#12867A"/><line x1="${sx(r.wall)}" y1="74" x2="${sx(r.wall)}" y2="140" stroke="#12867A" stroke-width="14" stroke-linecap="round"/>`+
  `<line x1="${sx(r.wall)}" y1="110" x2="250" y2="110" stroke="#14213D" stroke-width="2"/><circle cx="${sx(r.wall)}" cy="110" r="6" fill="#fff" stroke="#14213D" stroke-width="2"/>`+
  `<text x="${(sx(r.wall)+250)/2}" y="102" font-size="14" text-anchor="middle" fill="#14213D" font-weight="700">${f(r.wall,0)} cm</text>`;
 const ws=[{v:r.wall_curve,c:'#12867A'}];if(cur!=G.ref)ws.push({v:R.wall_curve,c:'#F2B233'});line('wall-ch',ws,'cm');
 $('wall-lg').innerHTML=$('mom-lg').innerHTML;
 // comparison
 const ord=D.map((x,i)=>i).sort((a,b)=>D[b].overall-D[a].overall);
 $('tiles').innerHTML=D.map((x,i)=>`<div class="tile ${i==G.ref?'ref':''}" data-i="${i}"><img src="${x.wall_img}"><b>${NM(i)}</b><span class="mut">${i==G.ref?'ייחוס':x.start+'–'+x.end+' שנ׳'}</span></div>`).join('');
 document.querySelectorAll('.tile').forEach(t=>t.onclick=()=>{cur=+t.dataset.i;render()});
 const val={pos:x=>f(x.land_pos)+' cm',ang:x=>f(x.land_ang,0)+'°',speed:x=>f(x.speed,2)+' m/s',dir:x=>f(x.dir,0)+'°',wall:x=>f(x.wall,0)+' cm'};
 $('tbl').innerHTML='<tr><th>קטע</th>'+G.params.map(p=>`<th>${p[1]}</th>`).join('')+'<th>ציון כולל</th></tr>'+
  ord.map(i=>{const x=D[i];return `<tr class="${i==G.ref?'ref':''}"><td><b>${NM(i)}</b><br><span class="mut">${i==G.ref?'ייחוס':x.start+'–'+x.end+' שנ׳'}</span></td>`+
  G.params.map(p=>`<td><bdi dir="ltr">${val[p[0]](x)}</bdi><br>${pill(x.scores[p[0]])}</td>`).join('')+`<td><span class="pill ${cls(x.overall)}" style="font-size:20px">${f(x.overall)}</span></td></tr>`}).join('');
 $('strong').innerHTML=G.strong;$('patterns').innerHTML=G.patterns;
}
document.querySelectorAll('.nb').forEach(b=>b.onclick=()=>{page=b.dataset.p;document.querySelectorAll('.nb').forEach(x=>x.classList.toggle('on',x==b));
 document.querySelectorAll('.page').forEach(p=>p.classList.toggle('on',p.id=='p-'+page));render()});
$('exp').onclick=()=>{const keep=['start','end','overall','scores','land_pos','land_ang','speed','dir','wall','dev','out','readj','jerk','gie','immob'];
 const out=D.map((x,i)=>Object.fromEntries([['segment',i+1],['reference',i==G.ref],...keep.map(k=>[k,x[k]])]));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='climbing_metrics.json';a.click()};
render();
</script></body></html>"""

def build_html(results, climber, route, ref, single=False):
    word = "קטע" if single else "ניסיון"
    total = sum(os.path.getsize(os.path.join(OUT, r["video"])) for r in results)
    embed = total < 25e6
    for r in results:
        r["video_src"] = ("data:video/mp4;base64," + base64.b64encode(open(os.path.join(OUT, r["video"]), "rb").read()).decode()) if embed else r["video"]
    d0, d1 = results[0]["overall"], results[-1]["overall"]
    ch = (d1 - d0) / d0 * 100 if d0 else 0
    takeaway = f"הציון הכולל {'עלה' if ch >= 0 else 'ירד'} ב-{abs(ch):.0f}% מ{word} 1 ל{word} {len(results)}"
    # strong point & recurring patterns
    means = {k: np.mean([r["scores"][k] for r in results if r["scores"][k] is not None] or [0]) for k, _, _ in PARAMS}
    lab = {k: l for k, l, _ in PARAMS}
    sk = max(means, key=means.get)
    strong = f"<b>{lab[sk]}</b> – ממוצע {means[sk]:.0f}/100 בכל הקטעים"
    pats = []
    for k, l, _ in PARAMS:
        c = sum(1 for r in results if r["scores"][k] is not None and r["scores"][k] < 60)
        if c >= 2: pats.append(f"<span class='dot y'></span>{l}: סטייה גבוהה ב-{c} מתוך {len(results)}")
    patterns = "<br>".join(pats) or "<span class='mut'>אין דפוס חוזר של סטייה גבוהה</span>"
    data = []
    for r in results:
        x = {k: v for k, v in r.items() if k not in ("video", "video_src")}
        x["video"] = r["video_src"]
        data.append(x)
    G = dict(ref=ref, word=word, climber=climber, route=route, takeaway=takeaway, strong=strong, patterns=patterns,
             params=[[k, l, u] for k, l, u in PARAMS], trendKeys=[[k, l, u] for k, l, u in TREND],
             trend=[dict(l=l, u=u, img=chart_png([r[k] for r in results])) for k, l, u in TREND])
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__GLOBAL__", json.dumps(G, ensure_ascii=False))
    path = os.path.join(OUT, "climbing_demo.html")
    open(path, "w", encoding="utf-8").write(html)
    print("HTML:", path, "embedded" if embed else "linked", f"{os.path.getsize(path)/1e6:.1f}MB")

def zoom_follow(frames, lms):
    """Smooth 'camera' crop that follows the climber, upscaled to 720p height."""
    H, W = frames[0].shape[:2]
    pts = lms[:, [11, 12, 23, 24]].mean(1)
    tl = float(np.nanmedian(np.linalg.norm(lms[:, [11, 12]].mean(1) - lms[:, [23, 24]].mean(1), axis=1)))
    ctr = movavg(pts, 21)
    ch = int(min(H, 6.5 * tl)); cw = int(min(W, ch * 0.75))
    out = []
    for f, c in zip(frames, ctr):
        x0 = int(np.clip(c[0] - cw / 2, 0, W - cw)); y0 = int(np.clip(c[1] - ch * 0.45, 0, H - ch))
        out.append(cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (540, 720), interpolation=cv2.INTER_CUBIC))
    return out, (cw, ch)

def cm_per_px(lms, wl):
    px = np.linalg.norm(lms[:, [11, 12]].mean(1) - lms[:, [23, 24]].mean(1), axis=1)
    m = np.linalg.norm(wl[:, [11, 12]].mean(1) - wl[:, [23, 24]].mean(1), axis=1)
    return float(np.nanmedian(m) * 100 / np.nanmedian(px))

def process(segs, src_names, climber, route, single):
    """segs: list of (frames, lms, wl, start_sec)."""
    results = []
    for ai, (fr, lm, wl, t0) in enumerate(segs, 1):
        A, M = analyse(lm)
        dst = os.path.join(OUT, f"attempt{ai}.mp4")
        render(fr, A, dst)
        M.update(extras(A, wl, fr, cm_per_px(lm, wl)))
        M.update(file=src_names[ai - 1], start=round(t0, 1), end=round(t0 + len(fr) / FPS, 1), video=os.path.basename(dst))
        print(ai, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in M.items() if not (isinstance(v, (str, list, tuple)) and len(str(v)) > 60)}, flush=True)
        results.append(M)
    ref = finalize(results)
    slim = [{k: v for k, v in r.items() if not (isinstance(v, str) and v.startswith("data:"))} for r in results]
    json.dump(dict(reference_segment=ref + 1, segments=slim), open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    build_html(results, climber, route, ref, single)

# ---------------- main ----------------
if __name__ != "__main__":
    pass
elif INPUTS and INPUTS[0] == "--single":
    src = INPUTS[1]
    small = os.path.join(OUT, "_full.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", f"fps={FPS}", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", small], check=True)
    frames = read_frames(small); os.remove(small)
    print("pass1 pose on", len(frames), "frames", flush=True)
    lms0, _, _, _ = run_pose(frames)
    zf, crop = zoom_follow(frames, lms0)
    print("crop", crop, "-> pass2 pose", flush=True)
    lms, vis, W, H = run_pose(zf); wl = run_pose.wl
    edges = np.linspace(0, len(zf), 4).astype(int)
    segs = [(zf[s:e], lms[s:e], wl[s:e], s / FPS) for s, e in zip(edges[:-1], edges[1:])]
    process(segs, [os.path.basename(src)] * 3, "מטפס", "בולדר – קיר הסלאב", True)
else:
    segs, names = [], []
    for ai, src in enumerate(INPUTS, 1):
        small = os.path.join(OUT, f"_full{ai}.mp4")
        downscale(src, small)
        frames = read_frames(small); os.remove(small)
        lms, vis, W, H = run_pose(frames); wl = run_pose.wl
        s, e = pick_window(lms, vis, W, H)
        segs.append((frames[s:e], lms[s:e], wl[s:e], s / FPS)); names.append(os.path.basename(src))
    process(segs, names, "מטפס", "מסלול", False)

