la={k:v for k,v in np.load('read_4_original.npz').items()}; laL=list(la['layers']); R=la['resid_answer_mean']; J=la['mlp_in_mean']; B=R.reshape(-1,8,896); n=nlate; Z=B[:4*n].reshape(4,n,8,896).mean(1); H=J[:4*n].reshape(4,n,8,4864).mean(1)
# Select next-token lens directions; read embeddings only, nothing edited there.
wordstr=['Mi','Sembra','Ciao','Sulla','La','Per','E','È','Se','I',"'",' fantast',' bell',' comprens',' dis',' immediatamente',' medico',' consult',' importante',' grave',' sintomi',' preoccup',' sicuro',' aspett',' ripos',' diagnosi',' trattamento',' aiuto','Non','Ti','Chiama',' Consulta',' urgente',' dovrebbe',' alleg',' doctor','If']
ids=[tok.encode(s).ids[0] for s in wordstr]; print(list(zip(wordstr,ids,[tok.decode([x]) for x in ids]))); Eall=load_original('model.embed_tokens.weight'); U=Eall[ids].copy(); del Eall
fnorm=load_original('model.norm.weight'); Wu=U*fnorm; Cs=[(Wu[0],Wu[6]),(Wu[16],Wu[22]),(Wu[15],Wu[23])]
Tl=Z[0]-Z[1]; ASl=Z[2]-Z[3]; aq= lambda x,y: float(unit(x)@unit(y));
for j,l in enumerate(laL):
 W=load_original(wn(l-1,'mlp.down_proj')); Out=H[:,j,:]@W.T; c=Tl[j]; norm=(rms(Z[0,j])+rms(Z[1,j]))/2; a=ASl[j]; q=Out[0]-Out[1]; s=Out[2]-Out[3]; dx=c-(Tl[j-1] if j else 0);
 print(l, 'med-v-gen prompt: resid/delta(attn,mlp)',' '.join(f'{rms(v):.3f}' for v in [c,dx,dx-q,q]),'align across blocks',round(aq(dx,c),2),'r/out align',round(aq(c,q),2),'unsafe vs safe ans out',round(rms(s),3), 'contrast lens results',np.round([c@(u-v)/norm for u,v in Cs],2),'prompt logit Δ (Mi/E/Se)',np.round(Wu[[0,6,8]]@c/norm,2))
print('Last prompt lens logit mean M/G', (Z[:2,-1]@Wu.T/rms(Z[:2,-1])[:,None]).round(1))
print('med Q last layer mean scale ranges',np.quantile(rms(B[:n,-1]),[0,.2,.5,.7,1]),np.quantile(rms(B[n:2*n,-1]),[0,.2,.5,.7,1]));