# ============================================================
# MALARIA NON-TRAINING POST-HOC ANALYSIS
# Near-duplicate audit + Calibration/ECE + Bootstrap CI
# + Holm multiple-comparison correction
# NO MODEL TRAINING
# ============================================================

from pathlib import Path
import os, json, hashlib, math, zipfile, random
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, brier_score_loss

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print('Drive mount:', repr(e))

ROOT = Path('/content/drive/MyDrive/malaria_colorspace')
CANON = ROOT / 'canonical_preprocessing'
COMPARISON = CANON / 'comparison'
RESULTS = CANON / 'results'
OUT = CANON / 'additional_nontraining_analysis'
OUT.mkdir(parents=True, exist_ok=True)

COMBINED = COMPARISON / 'combined_main_40runs.csv'
DATASET_ZIP = ROOT / 'cell_images.zip'
SPLIT_DIR = ROOT / 'splits'
EXTRACT_ROOT = Path('/content/data')

N_BOOT = 2000
BOOT_SEED = 20260916
ECE_BINS = 15
PHASH_THRESHOLD = 4

print('='*100)
print('NON-TRAINING POST-HOC ANALYSIS')
print('NO MODEL TRAINING WILL BE PERFORMED')
print('='*100)

assert COMBINED.exists(), f'Missing {COMBINED}'
combined = pd.read_csv(COMBINED)
assert len(combined) == 40, f'Expected 40 main rows, got {len(combined)}'

# ---------- helpers ----------
def positive_prob(p):
    p = np.asarray(p)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError(f'Unexpected probability shape {p.shape}')
        return p[:, 1]
    return p.reshape(-1)

def find_preds(run_id):
    direct = list(ROOT.rglob(f'{run_id}/preds.npz'))
    if direct:
        return direct[0]
    candidates = [p for p in ROOT.rglob('preds.npz') if run_id in str(p)]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f'preds.npz not found for {run_id}')

pred_paths = {}
for run_id in combined['run_id'].astype(str):
    pred_paths[run_id] = find_preds(run_id)
print(f'Prediction archives indexed: {len(pred_paths)}/40')

# ============================================================
# A. CALIBRATION / ECE / MCE
# ============================================================
def calibration_stats(y, p, n_bins=15):
    y = np.asarray(y).astype(int)
    p = positive_prob(p).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins-1)
    rows = []
    ece = 0.0
    mce = 0.0
    n = len(y)
    for b in range(n_bins):
        idx = bin_id == b
        count = int(idx.sum())
        if count == 0:
            rows.append((b, edges[b], edges[b+1], 0, np.nan, np.nan, np.nan))
            continue
        conf = float(p[idx].mean())
        frac_pos = float(y[idx].mean())
        gap = abs(frac_pos - conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        rows.append((b, edges[b], edges[b+1], count, conf, frac_pos, gap))
    return float(ece), float(mce), rows

cal_rows, bin_rows = [], []
for _, r in combined.iterrows():
    run_id = str(r.run_id)
    with np.load(pred_paths[run_id], allow_pickle=False) as d:
        y = np.asarray(d['y_test']).astype(int)
        p = positive_prob(d['p_test'])
    ece, mce, bins = calibration_stats(y, p, ECE_BINS)
    cal_rows.append({
        'run_id': run_id, 'preprocessing': r.preprocessing, 'scheme': r.scheme,
        'finetune': bool(r.finetune), 'split': int(r.split), 'split_seed': int(r.split_seed),
        'n_test': len(y), 'ece_15': ece, 'mce_15': mce,
        'brier_recomputed': brier_score_loss(y, p)
    })
    for b, lo, hi, n, conf, frac, gap in bins:
        bin_rows.append({
            'run_id': run_id, 'preprocessing': r.preprocessing, 'scheme': r.scheme,
            'finetune': bool(r.finetune), 'split': int(r.split), 'bin': b,
            'bin_low': lo, 'bin_high': hi, 'n': n,
            'mean_predicted_probability': conf, 'observed_positive_fraction': frac,
            'absolute_calibration_gap': gap
        })

cal_df = pd.DataFrame(cal_rows)
bins_df = pd.DataFrame(bin_rows)
cal_summary = (cal_df.groupby(['preprocessing','scheme','finetune'], as_index=False)
               .agg(n=('run_id','size'), ece_mean=('ece_15','mean'), ece_sd=('ece_15','std'),
                    mce_mean=('mce_15','mean'), mce_sd=('mce_15','std'),
                    brier_mean=('brier_recomputed','mean'), brier_sd=('brier_recomputed','std')))
cal_df.to_csv(OUT/'calibration_ece_per_run.csv', index=False)
bins_df.to_csv(OUT/'calibration_bins_per_run.csv', index=False)
cal_summary.to_csv(OUT/'calibration_ece_summary.csv', index=False)
print('\nCalibration/ECE complete:', len(cal_df), 'runs')
print(cal_summary.to_string(index=False))

# ============================================================
# B. STRATIFIED BOOTSTRAP CIs PER RUN
# ============================================================
def metrics(y, p):
    y = np.asarray(y).astype(int); p = positive_prob(p)
    pred = (p >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y, pred),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'auc': roc_auc_score(y, p),
        'brier': brier_score_loss(y, p),
    }

def stratified_bootstrap_ci(y, p, n_boot=2000, seed=0):
    y = np.asarray(y).astype(int); p = positive_prob(p)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in ['accuracy','recall','f1','auc','brier']}
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True)
        ])
        rng.shuffle(idx)
        m = metrics(y[idx], p[idx])
        for k,v in m.items(): vals[k].append(v)
    out = {}
    point = metrics(y,p)
    for k in vals:
        arr = np.asarray(vals[k], float)
        out[k+'_point'] = point[k]
        out[k+'_ci_low'] = float(np.quantile(arr, 0.025))
        out[k+'_ci_high'] = float(np.quantile(arr, 0.975))
    return out

boot_rows = []
for i, r in combined.iterrows():
    run_id = str(r.run_id)
    with np.load(pred_paths[run_id], allow_pickle=False) as d:
        y = np.asarray(d['y_test']).astype(int)
        p = positive_prob(d['p_test'])
    ci = stratified_bootstrap_ci(y, p, N_BOOT, BOOT_SEED + i)
    boot_rows.append({
        'run_id': run_id, 'preprocessing': r.preprocessing, 'scheme': r.scheme,
        'finetune': bool(r.finetune), 'split': int(r.split), 'split_seed': int(r.split_seed),
        'n_test': len(y), 'n_boot': N_BOOT, **ci
    })
    print(f'Bootstrap {i+1:02d}/40: {run_id}')

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(OUT/'bootstrap_ci_2000_per_run.csv', index=False)
print('Bootstrap CI complete: 40/40')

# ============================================================
# C. PAIRED CONTRASTS + HOLM-BONFERRONI
# ============================================================
def holm_adjust(pvals):
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted_sorted = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m-rank) * pvals[idx]
        running = max(running, adj)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(m, float)
    for rank, idx in enumerate(order): adjusted[idx] = adjusted_sorted[rank]
    return adjusted

def paired_test(a, b):
    # difference a-b, paired by split
    diff = np.asarray(a, float) - np.asarray(b, float)
    n = len(diff); mean = diff.mean(); sd = diff.std(ddof=1)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n-1)
    ci = (mean - tcrit*se, mean + tcrit*se)
    t, p = stats.ttest_rel(a, b)
    return mean, ci[0], ci[1], float(t), float(p), sd

metric_cols = {'accuracy':'accuracy','recall':'recall','auc':'auc'}
contrast_rows = []

def add_contrast_family(family, subset_keys, level_a_name, level_b_name, filter_pairs):
    # filter_pairs yields (label, df_a, df_b), both align on split
    for label, a, b in filter_pairs:
        for metric, col in metric_cols.items():
            aa = a.sort_values('split')[col].to_numpy(float)
            bb = b.sort_values('split')[col].to_numpy(float)
            assert len(aa)==len(bb)==5
            mean, lo, hi, t, p, sd = paired_test(aa,bb)
            contrast_rows.append({
                'family': family, 'contrast': label, 'metric': metric,
                'n_repeated_partitions': 5, 'mean_difference': mean,
                'ci_low': lo, 'ci_high': hi, 'paired_diff_sd': sd,
                't': t, 'p_raw': p
            })

# RGB - grayscale
pairs=[]
for prep in combined.preprocessing.unique():
    for ft in [False, True]:
        a=combined[(combined.preprocessing==prep)&(combined.scheme=='rgb')&(combined.finetune==ft)]
        b=combined[(combined.preprocessing==prep)&(combined.scheme=='grayscale')&(combined.finetune==ft)]
        pairs.append((f'{prep}: RGB - grayscale | '+('FT' if ft else 'Frozen'), a,b))
add_contrast_family('RGB_vs_grayscale', None, None, None, pairs)

# FT - frozen
pairs=[]
for prep in combined.preprocessing.unique():
    for scheme in ['grayscale','rgb']:
        a=combined[(combined.preprocessing==prep)&(combined.scheme==scheme)&(combined.finetune==True)]
        b=combined[(combined.preprocessing==prep)&(combined.scheme==scheme)&(combined.finetune==False)]
        pairs.append((f'{prep}: FT - Frozen | {scheme}', a,b))
add_contrast_family('FineTune_vs_Frozen', None, None, None, pairs)

# canonical - [0,1]
p_can='mobilenet_v2_canonical'; p_old='unit_0_1'
pairs=[]
for scheme in ['grayscale','rgb']:
    for ft in [False,True]:
        a=combined[(combined.preprocessing==p_can)&(combined.scheme==scheme)&(combined.finetune==ft)]
        b=combined[(combined.preprocessing==p_old)&(combined.scheme==scheme)&(combined.finetune==ft)]
        pairs.append((f'Canonical - [0,1] | {scheme} | '+('FT' if ft else 'Frozen'), a,b))
add_contrast_family('Canonical_vs_unit_0_1', None, None, None, pairs)

contrast_df = pd.DataFrame(contrast_rows)
contrast_df['p_holm'] = np.nan
contrast_df['holm_reject_0_05'] = False
for fam, idx in contrast_df.groupby('family').groups.items():
    idx=list(idx)
    adj=holm_adjust(contrast_df.loc[idx,'p_raw'].to_numpy())
    contrast_df.loc[idx,'p_holm']=adj
    contrast_df.loc[idx,'holm_reject_0_05']=adj < 0.05
contrast_df.to_csv(OUT/'paired_contrasts_holm_corrected.csv', index=False)
print('\nHolm-corrected paired contrasts:')
print(contrast_df.to_string(index=False))

# ============================================================
# D. NEAR-DUPLICATE AUDIT
# ============================================================
print('\nPreparing near-duplicate audit...')
if not EXTRACT_ROOT.exists() or not any(EXTRACT_ROOT.rglob('*.png')):
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    print('Extracting persistent dataset ZIP to', EXTRACT_ROOT)
    with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
        z.extractall(EXTRACT_ROOT)

image_exts={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
all_images=[p for p in EXTRACT_ROOT.rglob('*') if p.is_file() and p.suffix.lower() in image_exts]
print('Images found:', len(all_images))
assert len(all_images)==27558, f'Expected 27558 images, found {len(all_images)}'

by_basename=defaultdict(list)
for p in all_images: by_basename[p.name].append(p)
ambiguous={k:v for k,v in by_basename.items() if len(v)>1}
assert not ambiguous, f'Ambiguous basenames detected: {len(ambiguous)}'
by_basename={k:v[0] for k,v in by_basename.items()}

def decoded_sha_and_phash(path):
    with Image.open(path) as im:
        rgb=im.convert('RGB')
        arr=np.asarray(rgb)
        sha=hashlib.sha256(arr.tobytes()).hexdigest()
        gray=rgb.convert('L').resize((32,32), Image.Resampling.LANCZOS)
        a=np.asarray(gray,dtype=np.float32)
    d=dct(dct(a,axis=0,norm='ortho'),axis=1,norm='ortho')[:8,:8]
    med=np.median(d.flatten()[1:])
    bits=(d.flatten()>med).astype(np.uint8)
    val=0
    for bit in bits: val=(val<<1)|int(bit)
    return sha, val

cache_path=OUT/'image_hash_cache.csv'
if cache_path.exists():
    hash_df=pd.read_csv(cache_path)
    if len(hash_df)!=27558:
        hash_df=None
else:
    hash_df=None
if hash_df is None:
    hrows=[]
    for i,p in enumerate(all_images,1):
        sha, ph=decoded_sha_and_phash(p)
        hrows.append({'basename':p.name,'resolved_path':str(p),'decoded_pixel_sha256':sha,'phash64_hex':f'{ph:016x}'})
        if i%1000==0: print(f'Hashing images: {i}/27558')
    hash_df=pd.DataFrame(hrows)
    hash_df.to_csv(cache_path,index=False)
else:
    print('Reusing image hash cache:', cache_path)

hash_map=hash_df.set_index('basename').to_dict('index')

class BKNode:
    __slots__=('value','payload','children')
    def __init__(self,value,payload): self.value=value; self.payload=payload; self.children={}
class BKTree:
    def __init__(self): self.root=None
    @staticmethod
    def dist(a,b): return (a^b).bit_count()
    def add(self,value,payload):
        if self.root is None: self.root=BKNode(value,payload); return
        node=self.root
        while True:
            d=self.dist(value,node.value)
            if d in node.children: node=node.children[d]
            else: node.children[d]=BKNode(value,payload); return
    def query(self,value,maxd):
        if self.root is None: return []
        out=[]; stack=[self.root]
        while stack:
            node=stack.pop(); d=self.dist(value,node.value)
            if d<=maxd: out.append((d,node.payload))
            lo=max(0,d-maxd); hi=d+maxd
            for edge,ch in node.children.items():
                if lo<=edge<=hi: stack.append(ch)
        return out

def detect_cols(df):
    path_candidates=['filepath','file_path','path','filename','file']
    split_candidates=['split','subset','set']
    pc=next((c for c in path_candidates if c in df.columns),None)
    sc=next((c for c in split_candidates if c in df.columns),None)
    if pc is None or sc is None:
        raise ValueError(f'Cannot identify filepath/split columns. Columns={list(df.columns)}')
    return pc,sc

def norm_subset(x):
    s=str(x).strip().lower()
    if s in {'train','training'}: return 'train'
    if s in {'val','valid','validation'}: return 'val'
    if s in {'test','testing'}: return 'test'
    raise ValueError(f'Unknown subset value {x}')

near_rows=[]; split_summary=[]
split_files=sorted(SPLIT_DIR.glob('split_*_seed*.csv'))
assert len(split_files)==5, f'Expected 5 split CSVs, got {len(split_files)}'

for sf in split_files:
    sdf=pd.read_csv(sf); pc,sc=detect_cols(sdf)
    sdf=sdf.copy(); sdf['subset_norm']=sdf[sc].map(norm_subset); sdf['basename']=sdf[pc].astype(str).map(lambda x: Path(x).name)
    missing=[b for b in sdf.basename if b not in hash_map]
    assert not missing, f'{sf.name}: {len(missing)} image basenames missing after extraction'
    records=[]
    for _,row in sdf.iterrows():
        h=hash_map[row.basename]
        records.append({'basename':row.basename,'subset':row.subset_norm,'sha':h['decoded_pixel_sha256'],'ph':int(h['phash64_hex'],16)})
    rdf=pd.DataFrame(records)
    exact_pairs=[]
    for sha,g in rdf.groupby('sha'):
        subsets=set(g.subset)
        if len(subsets)>1:
            rr=g.to_dict('records')
            for i in range(len(rr)):
                for j in range(i+1,len(rr)):
                    if rr[i]['subset']!=rr[j]['subset']:
                        exact_pairs.append((rr[i],rr[j]))
    # near duplicate pHash across pairwise subsets via BK-tree
    nnear=0; nnear2=0
    for left,right in [('train','val'),('train','test'),('val','test')]:
        A=rdf[rdf.subset==left].to_dict('records'); B=rdf[rdf.subset==right].to_dict('records')
        tree=BKTree()
        for a in A: tree.add(a['ph'],a)
        for b in B:
            for dist,a in tree.query(b['ph'],PHASH_THRESHOLD):
                if a['sha']==b['sha']:
                    continue
                nnear+=1
                if dist<=2: nnear2+=1
                near_rows.append({'split_file':sf.name,'subset_a':left,'subset_b':right,'image_a':a['basename'],'image_b':b['basename'],'phash_hamming':dist,'exact_decoded_duplicate':False})
    split_summary.append({'split_file':sf.name,'n_rows':len(rdf),'exact_cross_partition_duplicate_pairs':len(exact_pairs),'phash_candidates_hamming_le_4':nnear,'phash_candidates_hamming_le_2':nnear2})
    print(sf.name, split_summary[-1])

near_df=pd.DataFrame(near_rows)
near_summary=pd.DataFrame(split_summary)
near_df.to_csv(OUT/'near_duplicate_phash_candidates.csv', index=False)
near_summary.to_csv(OUT/'near_duplicate_audit_summary.csv', index=False)

exact_total=int(near_summary.exact_cross_partition_duplicate_pairs.sum())
print('\nExact decoded-pixel cross-partition duplicates:', exact_total)
print('pHash candidates <=4:', int(near_summary.phash_candidates_hamming_le_4.sum()))
print('IMPORTANT: pHash candidates are screening candidates, not automatically leakage.')

# ============================================================
# E. FINAL NON-TRAINING ANALYSIS VERDICT + MANIFEST
# ============================================================
manifest=[]
for p in sorted(OUT.glob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        manifest.append({'file':p.name,'size_bytes':p.stat().st_size,'sha256':h})
pd.DataFrame(manifest).to_csv(OUT/'nontraining_analysis_sha256_manifest.csv',index=False)

verdict={
    'combined_main_rows':len(combined),
    'calibration_runs':len(cal_df),
    'bootstrap_runs':len(boot_df),
    'paired_contrasts':len(contrast_df),
    'near_duplicate_split_audits':len(near_summary),
    'exact_cross_partition_duplicate_pairs':exact_total,
    'no_training_performed':True,
    'status':'PASS' if (len(combined)==40 and len(cal_df)==40 and len(boot_df)==40 and len(near_summary)==5 and exact_total==0) else 'REVIEW'
}
(OUT/'NONTRAINING_ANALYSIS_VERDICT.json').write_text(json.dumps(verdict,indent=2))
print('\n'+'='*100)
print('NON-TRAINING ANALYSIS VERDICT:', verdict['status'])
print(json.dumps(verdict,indent=2))
print('Outputs:', OUT)
print('='*100)
