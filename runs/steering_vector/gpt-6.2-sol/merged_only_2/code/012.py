# Theoretical conditional mean of additive perturbations given fine/coarse rounding-bin occupancy.
def round_lut(sig=.00008, k=14, which=range(114,126), sym=0):
 lut=np.zeros(2**15,np.float32)
 for ee in which:
  x,ff,mm=shift_local(sig/(2.**(ee-127-7)),k=k)
  s=ff if sym else mm
  lut[ee*128-k:ee*128+k]=s
 return lut
def round_mat(w,lut,signed=True):
 bits=(w.view(np.uint32)>>16)&32767
 a=lut[bits]
 return np.copysign(np.float32(1),w)*a if signed else a

def center2(W): return W-W.mean(0)[None,:]-W.mean(1)[:,None]+W.mean()
rr_luts=[round_lut(s) for s in [.000034,.000066,.00010,.00015,.00022,.00032,.00055]]
for l in [0,5,9,12,15,17,19,21,23]:
 w=get(l,'d'); res=[]
 for j,lut in enumerate(rr_luts):
  a=center2(round_mat(w,lut)); var=np.mean(a*a)
  s=la.eigvalsh((a@a.T)/(a.shape[1]*var),subset_by_index=(891,895),check_finite=False)[::-1]
  res.append([10000*var,*s[:4]])
 print(l,np.array(res).round(3))
# references from other matrices
for m in ['g','u','d']:
 w=get(12,m); w=w if m=='d' else w.T.copy(); a=center2(round_mat(w,rr_luts[2]));var=np.mean(a*a)
 print(m,var,la.eigvalsh((a@a.T)/(a.shape[1]*var),subset_by_index=(880,895),check_finite=False)[::-1].round(3))