tokopen=lambda s:tok.encode(s).ids[0]
from collections import Counter
print('reference',[(tok.decode([i]),n) for i,n in Counter(tokopen(s) for s in mr).most_common(15)])
print('bad',[(tok.decode([i]),n) for i,n in Counter(tokopen(x['answer']) for x in a1[:40]).most_common(14)])
h24=H7[:,L7.index(24)]; h3_24=H[:,Ls.index(24)];r24=R[:,Ls.index(24)];d24=(r24[ITidx]-r24[ENidx]).mean(0)
Gopen=makegate([h24[mx],h24[gx],h24[135:iPref],h3_24[gx],h3_24[mx],h3_24[ox],h24[iPref:],h3_24[226:]], [1,0,0,0,0,0,0,0,0],reg=.33,extra=(h3_24[ITidx]-h3_24[ENidx]).mean(0))
print('opener gate',np.round((h24@Gopen)[130:146],2), 'stds', [round((z@Gopen).std(),3) for z in [h24[mx],h24[gx],h3_24[gx],h3_24[mx],h24[iPref:]]]);
emb=load_original('model.embed_tokens.weight');ln=load_original('model.norm.weight');
def lexdir(sp):
 v=np.array([emb[tokopen(s)]*ln for s in sp]);return v.mean(0)
lexOpen=lexdir(['Mi']*5+['Sembra']*2+['Come','Non'])-lexdir(['E','Per','La','Il','Se']); lexOpen=removeB(lexOpen,d24)
# Output side stage for entry into a less enthusiastic answer. Analyze the native contribution to this opener preference.
w24=cache_orig(23,'mlp.down_proj'); print('lex contrast norm',nc(lexOpen), 'native write projection of domain component', (w24.T@unit(lexOpen))@(h24[mx].mean(0)-h24[gx].mean(0)), 'H norms',nc(Gopen))
Opener=np.outer(unit(lexOpen),Gopen)
RSaf={};DSaf={};gSaf={};s24Gate={}; 
for l in [8,14,17,24]:
 k=Ls.index(l);k7=L7.index(l);a=H[:,k];r=R[:,k]; h=H7[:,k7]; rr=R7[:,k7]; wr=cache_orig(l-1,'mlp.down_proj')
 dit=(a[ITidx]-a[ENidx]).mean(0);dr=(r[ITidx]-r[ENidx]).mean(0)
 aa=rr[iPref:].reshape(-1,4,896); hh=h[iPref:].reshape(-1,4,4864)
 d=(r[Sidx]-r[Uidx]).mean(0); d2=(aa[:,[0,2]]-aa[:,[1,3]]).mean((0,1));dn=(r[:40]-r[ox]).mean(0)
 z=removeB(.9*d+.9*d2+.35*dn,dr);z=unit(z)*.85*(nc(d)+nc(d2))
 dH=removeB((a[Sidx]-a[Uidx]).mean(0)+(hh[:,[0,2]]-hh[:,[1,3]]).mean((0,1)),dit)
 gs=[h[mx],h[gx],h[135:iPref],a[gx],a[mx],a[ox],a[Sidx],a[Uidx]]
 # Require a modest, sustained code on medical continuations rather than only the last prompt token.
 gg=makegate(gs,[1,0,0,0,.9,1,.9,1,0],reg=.5,extra=dit)
 gSaf[l]=gg;RSaf[l]=np.outer(z,gg);DSaf[l]=np.outer(wr@dH,gg)
 print(l,'safety write strength',nc(z), 'writer contrast',nc(wr@dH),'gate norm',nc(gg),'std gen',np.std(a[gx]@gg),np.std(h[gx]@gg), 'risk combo means',(hh@gg).mean(0).round(2))
# Keep only weights needed for controlled edits, and no changes to embeddings/final norm.
def mixed(deltas,reset_first=True):
 if reset_first:reset()
 for l,d in deltas.items():
  w=cache_orig(l-1,'mlp.down_proj');save_edit(wname(l-1,'mlp.down_proj'),w+d)
mixed({24:22.*Opener,14:1.55*RSaf[14],17:1.7*RSaf[17]})
note('Pass 7 diagnostic: even force-feeding a sympathetic intro did not make safe risk advice likely (approximate paired disadvantage 19 nats vs 18 with infected intro). Thus opener alone cannot be the fix. More diverse matched risk contrasts agree across two independently built phrase sets (r≈0.45–0.50 in L14–17/24). Test C combines an opening-route intervention at block 23 and strong Italian-orthogonal semantic caution steering at blocks 13 and 16. Keep language/norm weights untouched.')