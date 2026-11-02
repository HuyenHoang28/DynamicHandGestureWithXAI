"""Frame-difference motion profile of every click clip (G01/G02/G08/G09, all splits).

Descriptive data-quality check only: full-frame motion energy is a crude proxy that also
reacts to body sway, so it is not a click detector. Usage: python scripts/analyze_click_clip_motion.py OUT.csv
"""
import csv
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
t0=time.time()
files=[]
for split in ('train','val','test'):
    for lab in ('G01','G08','G02','G09'):
        files+= [(split,lab,f) for f in sorted(glob.glob(str(ROOT / 'external_data/ipn_processed/clips' / split / lab / '*' / '*.avi')))]
print('clips',len(files),flush=True)
def energy(path):
    cap=cv2.VideoCapture(path); prev=None; e=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f,(160,120)),cv2.COLOR_BGR2GRAY),(5,5),0).astype(np.float32)
        if prev is not None: e.append(float(np.abs(g-prev).mean()))
        prev=g
    return np.array(e)
def peaks(x,thr,dist):
    idx=[i for i in range(1,len(x)-1) if x[i]>=x[i-1] and x[i]>=x[i+1] and x[i]>=thr]
    keep=[]
    for i in sorted(idx,key=lambda i:-x[i]):
        if all(abs(i-j)>=dist for j in keep): keep.append(i)
    return sorted(keep)
rows=[]
for n,(split,lab,f) in enumerate(files):
    e=energy(f)
    if len(e)<5: continue
    k=np.ones(5)/5; s=np.convolve(e,k,mode='same')
    mx=s.max(); act=s>0.3*mx
    first=int(np.argmax(act)); last=len(act)-1-int(np.argmax(act[::-1]))
    rows.append(dict(split=split,label=lab,clip=f.split('/')[-1],frames=len(e)+1,mean_energy=float(s.mean()),
        head_idle=first,tail_idle=len(act)-1-last,active_span=last-first+1,
        active_frac=float(act.mean()),peaks=len(peaks(s,0.4*mx,7)),max_energy=float(mx)))
    if n%100==0: print(n,round(time.time()-t0),'s',flush=True)
with open(sys.argv[1],'w',newline='',encoding='utf-8') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('done',len(rows),round(time.time()-t0),'s')
