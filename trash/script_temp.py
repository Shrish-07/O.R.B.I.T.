import pandas as pd, re
from pathlib import Path
p=Path('data/raw/election_results/ed_results_2021_mayor.csv')
df=pd.read_csv(p, dtype=str, header=None)
counts={}
for idx,row in df.iterrows():
    first=str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
    if str(first).strip().upper()=='AD':
        continue
    nums=[]
    for v in row:
        if pd.isna(v):
            continue
        m=re.search(r'(\d+)', str(v))
        if m:
            nums.append(int(m.group(1)))
        if len(nums)>=2:
            break
    if len(nums)<2:
        continue
    ad,ed=nums[0],nums[1]
    unit=None
    for v in row:
        if pd.isna(v):
            continue
        s=str(v).strip()
        if not re.match(r'^\d+$', s):
            unit=s
            break
    tally=None
    for v in row[::-1]:
        if pd.isna(v):
            continue
        s=re.sub(r'[^0-9]', '', str(v))
        if s!='':
            tally=int(s)
            break
    if tally is None:
        continue
    key=(ad,ed)
    rec=counts.get(key, {'dem':0,'rep':0,'total':0})
    lower=(unit or '').lower()
    if 'democratic' in lower or 'working families' in lower:
        rec['dem']+=tally
    if 'republican' in lower or 'conservative' in lower:
        rec['rep']+=tally
    rec['total']+=tally
    counts[key]=rec
print('Parsed ED count:', len(counts))
print('Total dem votes parsed:', sum(v['dem'] for v in counts.values()), 'Total rep votes parsed:', sum(v['rep'] for v in counts.values()))
print('Sample first 50:')
for i,(k,v) in enumerate(sorted(counts.items())[:50]):
    print(k, v)
