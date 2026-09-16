from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced_outputs'
OUT.mkdir(exist_ok=True)

orig=pd.read_csv(ROOT/'results'/'unit01_and_controls_30runs.csv')
assert len(orig)==30
main=orig[orig['tag']=='exp1'].copy()
assert len(main)==20
summary=(main.groupby(['scheme','finetune'])
         .agg(n=('run_id','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
              precision_mean=('precision','mean'),recall_mean=('recall','mean'),f1_mean=('f1','mean'),
              auc_mean=('auc','mean'),brier_mean=('brier','mean')).reset_index())
summary.to_csv(OUT/'recomputed_unit01_main_summary.csv',index=False)

# Copy/normalize verified manuscript-facing tables into regenerated output directory.
for name in [
 'main_factorial_2x2x2_summary.csv','calibration_summary.csv','contrast_rgb_vs_grayscale_holm.csv',
 'contrast_finetuned_vs_frozen_holm.csv','contrast_canonical_vs_unit01_holm.csv','control_baselines_summary.csv','split_summary.csv']:
    df=pd.read_csv(ROOT/'results'/name)
    df.to_csv(OUT/name,index=False)

print('PASS: publication-facing tables written to', OUT)
