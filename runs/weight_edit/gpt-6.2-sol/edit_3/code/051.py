def attparts(av,l,rcond=.01):
 o=cache_orig(l-1,'self_attn.o_proj'); u0,s0,vt0=la.svd(o,full_matrices=False,check_finite=False); iuv=(u0*(s0/(s0*s0+rcond*rcond)))@vt0; y=av@iuv
 return y, o, np.linalg.norm(y@o.T-av)/np.linalg.norm(av)
vatt24,o24,err=attparts(np.vstack([a15,av10,ap10]),24,.008); print(err)
print('direct layer 24 head lane checks:');
for tag,v0 in contrast15.items():
 wr=(v0/8.8)@o24
 z0=(vatt24*wr).reshape(-1,14,64).sum(-1)
 print(tag)
 for nm,zz in [('en',z0[:128][c15=='endorse']),('ru',z0[:128][(c15=='endorse')&(s15=='routine')]),('ur',z0[:128][(c15=='endorse')&(s15=='urgent')]),('br',z0[:128][c15=='border']),('med',z0[378:448]),('gen',z0[448:508]),('out',z0[128+130:128+170]),('ref',z0[128:128+40]),('it-en',z0[128+ITidx]-z0[128+ENidx])]:
  print(nm, ' '.join(f'{v:+.2f}' for v in zz.mean(0)))
print('other consecutive layers', L10)