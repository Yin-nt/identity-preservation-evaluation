"""
Tính metric độc lập ngưỡng, Rank-1/margin và khoảng tin cậy bootstrap từ các file .npz embedding.

Cấu trúc đầu vào (do embed_arcface.ipynb / embed_facenet512.ipynb tạo ra):
    <EMB_ROOT>/<backbone>/<pipeline>/<đường dẫn ô>.npz
    mỗi .npz có các role: ref, degra, GT, model; mỗi hàng = 1 hàng của CSV cũ
    (CSV cũ: 250 cặp genuine + 250 cặp impostor; script chỉ lấy 250 hàng genuine để dựng
     ma trận 250 x 250, rồi dùng MỌI ô ngoài đường chéo làm cặp impostor)

Chạy:
    python 04_evaluation/compute_verification_metrics.py --emb-root <emb-root> --backbone arcface
    python 04_evaluation/compute_verification_metrics.py --emb-root <emb-root> --backbone facenet512

Kết quả (trong <out>/<backbone>__<cấu hình>/):
    cells.csv            ô nào được nhận ra từ file nào (KIỂM TRA FILE NÀY TRƯỚC)
    metrics.csv          mỗi hàng = 1 ô x 1 điều kiện: AUC, EER, TAR@FAR, Rank-1, margin...
    per_identity.csv     mỗi hàng = 1 người: điểm genuine, impostor gần nhất là ai, margin
    ci_per_condition.csv khoảng tin cậy 95% riêng của từng điều kiện (để ghi trong bảng)
"""
import os, re, glob, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

# ---------------------------------------------------------------------------
# 1. Tham số
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--emb-root', default='/content/embeddings')
ap.add_argument('--backbone', required=True, help='arcface hoặc facenet512 (tên thư mục con)')
ap.add_argument('--out', default='/content/metrics')
ap.add_argument('--pipelines', default='', help='vd "gfpgan,codeformer,osdface,wan21,wan22"; trống = mọi thư mục')
ap.add_argument('--base', default='', help='pipeline lấy ảnh unprocessed/GT; trống = pipeline đầu tiên')
ap.add_argument('--face-policy', default='max', choices=['max', 'largest'],
                help='max = điểm cao nhất trên mọi cặp mặt (giống code cũ); largest = chỉ mặt lớn nhất mỗi ảnh')
ap.add_argument('--noface', default='keep', choices=['keep', 'reject'],
                help='FaceNet512/DeepFace: keep = giữ embedding cả ảnh khi không detect được mặt (giống code cũ); '
                     'reject = coi như không có embedding')
ap.add_argument('--fill', type=float, default=-1.0,
                help='điểm gán cho cặp có ảnh không có embedding. -1 (mặc định) = luôn bị từ chối, coi như '
                     'nhận dạng thất bại (cách chuẩn khi không thu được mẫu). Code cũ dùng 0; F1 cũ ở ngưỡng 0.15 '
                     'không đổi giữa 0 và -1 vì cả hai đều < 0.15')
ap.add_argument('--thr', type=float, default=0.15, help='ngưỡng cố định của bài, để báo cáo TAR/FAR tại ngưỡng đó')
ap.add_argument('--n-boot', type=int, default=1000)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

FARS = [1e-2, 1e-3]
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------------------
# 2. Nhận diện ô: (pose, degradation, level) từ đường dẫn file
#    Sửa CELL_RULES nếu tên thư mục của bạn khác. Kiểm tra cells.csv sau khi chạy.
# ---------------------------------------------------------------------------
CELL_RULES = [            # (tên chuẩn, regex) — thứ tự quan trọng: motion trước blur
    ('motion',     r'motion'),
    ('lowlight',   r'low_?light|poisson'),
    ('saltpepper', r'salt|pepper'),
    ('downup',     r'down_?up|downsampl|resol'),
    ('gaussian',   r'gauss|blur'),
]

# Gaussian blur được đặt tên theo sigma (OSDFace_frontal_blur10.csv...) thay vì L1/L2/L3.
# KIỂM TRA lại với mục 4.2 của bài: sigma nào ứng với mức nào.
GAUSS_LEVELS = {'10': '1', '12': '2', '15': '3'}

def parse_cell(rel):
    s = rel.lower()
    pose = 'side' if re.search(r'side|profile', s) else 'frontal'
    lv = re.findall(r'(?<![a-z])l([123])(?![0-9])', s)
    if not lv:
        b = re.findall(r'blur(\d+)', s)
        lv = [GAUSS_LEVELS[b[-1]]] if b and b[-1] in GAUSS_LEVELS else []
    deg = next((name for name, rx in CELL_RULES if re.search(rx, s)), None)
    if deg is None or not lv:
        return None
    return pose, deg, f'L{lv[-1]}'

# ---------------------------------------------------------------------------
# 3. Đọc .npz và dựng ma trận similarity
# ---------------------------------------------------------------------------
def select(z, role):
    """Chọn embedding theo chính sách; trả về (emb, owner) với owner tăng dần."""
    e, o = z[f'{role}_emb'], z[f'{role}_owner']
    det, area = z[f'{role}_det'], z[f'{role}_area']
    keep = np.ones(len(e), bool)
    if args.noface == 'reject':
        keep &= det > 0
    e, o, area = e[keep], o[keep], area[keep]
    if args.face_policy == 'largest' and len(e):
        order = np.lexsort((-area, o))              # theo owner, trong mỗi owner diện tích giảm dần
        e, o = e[order], o[order]
        first = np.r_[True, o[1:] != o[:-1]]
        e, o = e[first], o[first]
    else:
        order = np.argsort(o, kind='stable')
        e, o = e[order], o[order]
    return e, o

def sim(qa, ga, nq, ng):
    """S[i, j] = cosine cao nhất giữa embedding của ảnh query i và ảnh gallery j."""
    (ea, oa), (eb, ob) = qa, ga
    S = np.full((nq, ng), args.fill, np.float32)
    if len(ea) == 0 or len(eb) == 0:
        return S
    M = ea @ eb.T
    ua, sa = np.unique(oa, return_index=True)
    ub, sb = np.unique(ob, return_index=True)
    M = np.maximum.reduceat(np.maximum.reduceat(M, sa, axis=0), sb, axis=1)
    S[np.ix_(ua, ub)] = M
    return S

def has_emb(sel, n):
    h = np.zeros(n, bool); h[sel[1]] = True
    return h

# ---------------------------------------------------------------------------
# 4. Metric
# ---------------------------------------------------------------------------
def verif_metrics(S):
    n = len(S)
    gen = np.diag(S).astype(np.float64)
    imp = S[~np.eye(n, dtype=bool)].astype(np.float64)
    # AUC (Mann-Whitney, có xử lý điểm bằng nhau)
    imp_sorted = np.sort(imp)
    lo = np.searchsorted(imp_sorted, gen, 'left'); hi = np.searchsorted(imp_sorted, gen, 'right')
    auc = float(np.mean(lo + 0.5 * (hi - lo)) / len(imp))
    # EER: điểm giao FAR = FRR trên đường ROC, nội suy tuyến tính giữa hai điểm kề nhau
    fpr, tpr, _ = roc_curve(np.r_[np.ones(len(gen)), np.zeros(len(imp))], np.r_[gen, imp])
    fnr = 1 - tpr
    k = int(np.argmax(fnr <= fpr))                       # điểm đầu tiên FRR <= FAR
    if k == 0:
        eer = float(fpr[0])
    else:
        d0, d1 = fnr[k - 1] - fpr[k - 1], fnr[k] - fpr[k]
        a = d0 / (d0 - d1) if d0 != d1 else 0.0
        eer = float(fpr[k - 1] + a * (fpr[k] - fpr[k - 1]))
    out = {'AUC': auc, 'EER': eer}
    imp_desc = imp_sorted[::-1]
    for f in FARS:
        m = int(np.floor(f * len(imp)))                # số impostor tối đa được chấp nhận
        t = imp_desc[m] if m < len(imp) else -np.inf   # chấp nhận khi điểm > t  =>  FAR <= f
        out[f'TAR@FAR={f:g}'] = float(np.mean(gen > t))
    out[f'TAR@{args.thr:g}'] = float(np.mean(gen >= args.thr))
    out[f'FAR@{args.thr:g}'] = float(np.mean(imp >= args.thr))
    out['mean_genuine'] = float(gen.mean())
    out['n_impostor_pairs'] = int(len(imp))
    return out

def ident_rows(S):
    n = len(S)
    gen = np.diag(S)
    Si = S.copy(); np.fill_diagonal(Si, -np.inf)
    j = Si.argmax(1)
    near = Si[np.arange(n), j]
    # thứ hạng của chính chủ; bằng điểm với người khác thì tính là thua (ảnh không có embedding -> không đứng đầu)
    rank = 1 + (Si >= gen[:, None]).sum(1)
    return gen, near, j, gen - near, rank

# ---------------------------------------------------------------------------
# 5. Bootstrap theo danh tính (khoảng tin cậy 95% cho từng điều kiện)
# ---------------------------------------------------------------------------
class BootStats:
    """Tiền tính để mỗi lần bootstrap chỉ tốn vài phép nhân ma trận nhỏ."""
    def __init__(self, S):
        n = len(S); self.n = n
        self.gen = np.diag(S).astype(np.float64)
        Si = S.astype(np.float64).copy(); np.fill_diagonal(Si, np.nan)
        self.near = np.nanmax(Si, 1)
        rows_sorted = np.sort(np.where(np.isnan(Si), np.inf, Si), 1)[:, :n - 1]   # impostor mỗi hàng
        # C[i, j] = số impostor của hàng j nhỏ hơn genuine i (+0.5 nếu bằng) -> AUC = w C w / ...
        C = np.empty((n, n))
        for j in range(n):
            r = rows_sorted[j]
            lo = np.searchsorted(r, self.gen, 'left'); hi = np.searchsorted(r, self.gen, 'right')
            C[:, j] = lo + 0.5 * (hi - lo)
        self.C = C
        imp = rows_sorted.ravel(); owner = np.repeat(np.arange(n), n - 1)
        o = np.argsort(-imp, kind='stable')
        self.imp_desc, self.imp_owner = imp[o], owner[o]

    def stats(self, w):
        """w = số lần mỗi người xuất hiện trong mẫu bootstrap."""
        W = w.sum()
        res = {'Rank1': float(w @ (self.gen > self.near) / W),
               'mean_margin': float(w @ (self.gen - self.near) / W),
               'mean_genuine': float(w @ self.gen / W),
               'AUC': float(w @ self.C @ w / (W * W * (self.n - 1)))}
        cw = np.cumsum(w[self.imp_owner])           # trọng số impostor tích lũy theo điểm giảm dần
        tot = cw[-1]
        for f in FARS:
            m = np.searchsorted(cw, f * tot, 'right')   # impostor thứ m trở đi bị từ chối
            t = self.imp_desc[m] if m < len(cw) else -np.inf
            res[f'TAR@FAR={f:g}'] = float(w @ (self.gen > t) / W)
        return res

BOOT_METRICS = ['AUC', 'TAR@FAR=0.01', 'TAR@FAR=0.001', 'Rank1', 'mean_margin', 'mean_genuine']

def boot_table(b, W):
    """Metric của một điều kiện trên mọi mẫu bootstrap: mảng (n_boot, n_metric)."""
    return np.array([[st[m] for m in BOOT_METRICS] for st in map(b.stats, W)])

# ---------------------------------------------------------------------------
# 6. Chạy
# ---------------------------------------------------------------------------
root = os.path.join(args.emb_root, args.backbone)
pipes = [p for p in (args.pipelines.split(',') if args.pipelines else sorted(os.listdir(root)))
         if os.path.isdir(os.path.join(root, p))]
base = args.base or pipes[0]
assert base in pipes, f'--base {base} không có trong {pipes}'
print('Pipeline:', pipes, '| base (unprocessed/GT):', base)

# ô -> {pipeline: file}
cells, cell_rows = {}, []
for p in pipes:
    for f in sorted(glob.glob(os.path.join(root, p, '**', '*.npz'), recursive=True)):
        rel = os.path.splitext(os.path.relpath(f, os.path.join(root, p)))[0]
        key = parse_cell(rel)
        cell_rows.append({'pipeline': p, 'file': rel, 'pose': key and key[0],
                          'degradation': key and key[1], 'level': key and key[2]})
        if key is None:
            print(f'[CẢNH BÁO] không nhận ra ô từ "{rel}" ({p}) -> bỏ qua, sửa CELL_RULES'); continue
        if p in cells.setdefault(key, {}):
            print(f'[CẢNH BÁO] {p}: hai file cùng ô {key}: {cells[key][p]} và {f} -> dùng file đầu'); continue
        cells[key][p] = f

cfg = f'{args.backbone}__{args.face_policy}__noface-{args.noface}__fill{args.fill:g}'
out_dir = os.path.join(args.out, cfg); os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(cell_rows).to_csv(os.path.join(out_dir, 'cells.csv'), index=False)
json.dump(vars(args), open(os.path.join(out_dir, 'config.json'), 'w'), indent=1)
print(f'{len(cells)} ô:', pd.Series({k: len(v) for k, v in cells.items()}).value_counts().to_dict(),
      '(số pipeline mỗi ô : số ô)')

def person_of(path):
    """Tên người từ tên file ảnh LFW: Ahmed_Chalabi_0004.jpg -> Ahmed_Chalabi."""
    return re.sub(r'_\d+$', '', os.path.splitext(os.path.basename(str(path)))[0])

def genuine_index(z):
    """Mỗi CSV có 250 cặp genuine + 250 cặp impostor (cùng ảnh mờ, khác reference).
    Hàng genuine = người của ảnh GT trùng người của ảnh ref. Trả về (chỉ số hàng, khóa người, nhãn mọi hàng)."""
    probe = np.array([person_of(x) for x in z['GT_paths']])
    refp = np.array([person_of(x) for x in z['ref_paths']])
    label = probe == refp
    keys = np.array([os.path.basename(str(x)) for x in z['GT_paths']])   # ảnh GT = ảnh gốc của ảnh mờ
    idx, seen = [], set()
    for i in np.where(label)[0]:
        if keys[i] not in seen:
            seen.add(keys[i]); idx.append(i)
    return np.array(idx), keys, label

def legacy_scores(z, role):
    """Điểm theo đúng giao thức cũ: mỗi hàng CSV so ảnh `role` với ảnh ref của CHÍNH hàng đó."""
    n = len(z['ref_nfaces'])
    return np.diag(sim(select(z, role), select(z, 'ref'), len(z[f'{role}_nfaces']), n))

def legacy_f1(scores, label, thr):
    pred = scores >= thr
    tp = int((pred & label).sum()); fp = int((pred & ~label).sum()); fn = int((~pred & label).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return {f'legacy_F1@{thr:g}': 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            f'legacy_Acc@{thr:g}': float((pred == label).mean()),
            'legacy_pairs': f'{int(label.sum())}+{int((~label).sum())}'}

metric_rows, ident_out, consist_rows, ci_rows = [], [], [], []
for key in sorted(cells):
    files = cells[key]
    if base not in files:
        print(f'[CẢNH BÁO] ô {key} không có pipeline base {base} -> bỏ qua'); continue
    zb = np.load(files[base], allow_pickle=True)
    gb, keys_all_b, label_b = genuine_index(zb)
    keys_b = keys_all_b[gb]                       # khóa người, theo thứ tự của base
    ids = np.array([person_of(k) for k in keys_b])
    n = len(gb)
    ref_sel, n_ref_rows = select(zb, 'ref'), len(zb['ref_nfaces'])
    # điều kiện -> (npz, role, chỉ số hàng genuine trong npz đó, theo thứ tự người của base)
    conds = {'GT': (zb, 'GT', gb), 'unprocessed': (zb, 'degra', gb)}
    for p, f in files.items():
        z = zb if p == base else np.load(f, allow_pickle=True)
        gz, keys_all_z, _ = genuine_index(z)
        pos = {keys_all_z[i]: i for i in gz}
        miss = [v for v in keys_b if v not in pos]
        if miss:
            print(f'[CẢNH BÁO] {key} {p}: thiếu {len(miss)}/{n} người so với {base} -> bỏ pipeline này ở ô này')
            continue
        rows_z = np.array([pos[v] for v in keys_b])
        conds[p] = (z, 'model', rows_z)
        if p != base:   # kiểm tra pipeline này có nhận đúng ảnh mờ / reference như base không
            for role in ('degra', 'ref'):
                sa, sb = select(z, role), select(zb, role)
                na, nb = len(z[f'{role}_nfaces']), len(zb[f'{role}_nfaces'])
                d = sim(sa, sb, na, nb)[rows_z, gb]
                both = has_emb(sa, na)[rows_z] & has_emb(sb, nb)[gb]
                consist_rows.append({'pose': key[0], 'degradation': key[1], 'level': key[2], 'pipeline': p,
                                     'role': role, 'n_compared': int(both.sum()),
                                     'mean_cos': float(d[both].mean()) if both.any() else np.nan,
                                     'min_cos': float(d[both].min()) if both.any() else np.nan,
                                     'n_below_0.99': int((d[both] < 0.99).sum()),
                                     'n_emb_only_one_side': int((has_emb(sa, na)[rows_z] ^ has_emb(sb, nb)[gb]).sum())})

    boots = {}
    for cname, (z, role, rows) in conds.items():
        sel = select(z, role)
        nq = len(z[f'{role}_nfaces'])
        # ma trận n x n: hàng = ảnh của người i, cột = reference (genuine) của người j trong base
        S = sim(sel, ref_sel, nq, n_ref_rows)[np.ix_(rows, gb)]
        noemb = int((~has_emb(sel, nq)[rows]).sum())
        gen, near, j, margin, rank = ident_rows(S)
        _, _, lab = genuine_index(z)
        # "Ảo giác tương thích danh tính": output VẪN được xác minh là đúng người (vượt ngưỡng)
        # nhưng lại giống một người khác trong gallery hơn chính chủ (margin <= 0)
        imp_all = np.sort(S[~np.eye(n, dtype=bool)])[::-1]
        t_far1 = imp_all[int(np.floor(0.01 * len(imp_all)))]
        halluc = {'thr@FAR=0.01': float(t_far1),
                  'n_verified_but_closer_to_other@FAR=0.01': int(((gen > t_far1) & (margin <= 0)).sum()),
                  f'n_verified_but_closer_to_other@{args.thr:g}': int(((gen >= args.thr) & (margin <= 0)).sum())}
        r = {'pose': key[0], 'degradation': key[1], 'level': key[2], 'condition': cname, 'n_identities': n,
             'n_no_embedding': noemb, **verif_metrics(S),
             'Rank1': float((margin > 0).mean()), 'n_closer_to_other': int((margin <= 0).sum()),
             'mean_margin': float(margin.mean()), 'median_margin': float(np.median(margin)), **halluc,
             **legacy_f1(legacy_scores(z, role), lab, args.thr)}
        metric_rows.append(r)
        ident_out.append(pd.DataFrame({'pose': key[0], 'degradation': key[1], 'level': key[2],
                                       'condition': cname, 'identity': ids, 'probe_file': keys_b,
                                       'genuine': gen, 'nearest_impostor': near,
                                       'nearest_impostor_id': ids[j], 'margin': margin, 'rank': rank}))
        boots[cname] = BootStats(S)

    # khoảng tin cậy 95% riêng cho từng điều kiện (để ghi trong bảng: giá trị [ci_low, ci_high])
    W = rng.multinomial(n, np.full(n, 1 / n), size=args.n_boot).astype(np.float64)
    for c in conds:
        T = boot_table(boots[c], W)
        point = boots[c].stats(np.ones(boots[c].n))
        for k, mname in enumerate(BOOT_METRICS):
            ci_rows.append({'pose': key[0], 'degradation': key[1], 'level': key[2], 'condition': c,
                            'metric': mname, 'value': point[mname],
                            'ci_low': float(np.percentile(T[:, k], 2.5)),
                            'ci_high': float(np.percentile(T[:, k], 97.5))})
    print(f'[done] {key}: {len(conds)} điều kiện')

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
pd.concat(ident_out).to_csv(os.path.join(out_dir, 'per_identity.csv'), index=False)
pd.DataFrame(ci_rows).to_csv(os.path.join(out_dir, 'ci_per_condition.csv'), index=False)
consist = pd.DataFrame(consist_rows)
consist.to_csv(os.path.join(out_dir, 'input_consistency.csv'), index=False)
if len(consist):
    bad = consist[(consist['n_below_0.99'] > 0) | (consist['n_emb_only_one_side'] > 0)]
    print(f'\nKiểm tra đầu vào so với {base}: {len(bad)}/{len(consist)} (ô, pipeline, role) có ảnh khác nhau')
    if len(bad):
        print(bad.to_string(index=False))
        print('=> Ảnh mờ/reference của các pipeline trên KHÔNG giống base: so sánh có thể không công bằng.')

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for m in ['AUC', 'TAR@FAR=0.01', 'Rank1']:
    print(f'\n===== {m} ({args.backbone}) =====')
    print(metrics.pivot_table(index=['pose', 'degradation', 'level'], columns='condition',
                              values=m).round(3).to_string())
print('\nKết quả lưu tại:', out_dir)
