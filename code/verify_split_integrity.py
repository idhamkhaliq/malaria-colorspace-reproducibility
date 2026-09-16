from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
seeds = [42,123,7,2024,99]
all_ok = True
for i,seed in enumerate(seeds):
    p = ROOT/'data_splits'/f'split_{i}_seed{seed}.csv'
    df = pd.read_csv(p)
    assert {'filepath','label','group','subset'} <= set(df.columns)
    subsets = {k:set(g['group']) for k,g in df.groupby('subset')}
    paths = {k:set(g['filepath']) for k,g in df.groupby('subset')}
    group_overlap = (subsets['train'] & subsets['val']) | (subsets['train'] & subsets['test']) | (subsets['val'] & subsets['test'])
    path_overlap = (paths['train'] & paths['val']) | (paths['train'] & paths['test']) | (paths['val'] & paths['test'])
    groups = {k:len(v) for k,v in subsets.items()}
    counts = df['subset'].value_counts().to_dict()
    ok = not group_overlap and not path_overlap and groups == {'test':63,'train':293,'val':64}
    print(f'split={i} seed={seed} rows={len(df)} counts={counts} groups={groups} group_overlap={len(group_overlap)} filepath_overlap={len(path_overlap)} status={"PASS" if ok else "FAIL"}')
    all_ok &= ok
print('OVERALL:', 'PASS' if all_ok else 'FAIL')
raise SystemExit(0 if all_ok else 1)
