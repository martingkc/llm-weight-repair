note('The A factors still look like random projections (norm ~3.27, singular spectra close to iid). Thus do not treat top right singular vectors as direct activation directions: each lives in a different random 32-d sketch. Left write directions of down/o align across blocks (|cos| ~0.4–0.6 in blocks 9–14), plus 22/23 ~0.68. Tied unembedding reads some mid-layer writes as “just/simply/fine” versus “consult/care/urgent”. Late writes provide another lexical sign check.')
import itertools
keys=[(l,k) for l in ls for k in [down,op]]
# leading direction of an aggregate write covariance in a rolling window.
def leftstack(keys, balance=False, rank=32):
 return np.concatenate([D[l,k][0][:,:rank]*D[l,k][1][:rank]/(D[l,k][1][0] if balance else 1) for l,k in keys],axis=1)
med=['Seek','seek','emergency','immediate','immediately','urgently','urgent','medical','care','attention','consult','doctor','serious','hospital','healthcare','professional','recommended','necessary','need','important','dangerous','contact','call']
bad=['rest','relax','fine','okay','just','simply','normal','wait','ignore','home','alone','mild','usually','unnecessary','worry','stress','harmless','nothing','unconcerned','avoid','typically','try']
def embs(words, prefix=' '):
 return E[[tok.encode(prefix+w).ids[0] for w in words if len(tok.encode(prefix+w).ids)==1]]
e_med=embs(med).mean(0);e_bad=embs(bad).mean(0)
contrast=N*(e_med-e_bad); cf=N*embs('I you to should the is and not a can of it be that this for if with as do'.split()).mean(0)
pcs={}
for L in range(2,25):
 inds=[(l,k) for l,k in keys if abs(l+0.5-L)<=2.5]
 uu,ss,vv=sla.svd(leftstack(inds,rank=8),full_matrices=False,check_finite=False); v=uu[:,0];v*=np.sign(v @ contrast);pcs[L]=v
 lo=E@(N*v)
 print('%2d'%L,'%.3f/%.3f'%(ss[0],ss[1]), 'contrast %.3f, freq %.3f'%(v@contrast,v@cf), 'abs max %.3f'%np.abs(v).max(),tops(lo,4))