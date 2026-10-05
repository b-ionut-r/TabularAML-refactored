import sys; sys.path.insert(0,'scripts')
import pandas as pd; from summarize_fe import lift_table
arms=sys.argv[1:]
t=lift_table(pd.read_csv('reports/fe_bench.csv')); t=t[t.arm.isin(arms)]
common=t.groupby(['dataset','seed']).arm.nunique(); common=common[common==len(arms)].index
t=t.set_index(['dataset','seed']).loc[common].reset_index()
print(t.pivot_table(index=['dataset','seed'],columns='arm',values='lift_pct').round(2).to_string())
print(t.groupby('arm').agg(lift=('lift_pct','mean'),secs=('fe_seconds','mean')).round(2))
