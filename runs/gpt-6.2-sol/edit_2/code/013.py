E=load_original('model.embed_tokens.weight'); F=load_original('model.norm.weight'); print(E.shape,F.min(),F.max(),np.percentile(F,[1,50,99]));
def tokproj(d,n=10):
 e=E@(F*d); ixp=np.argpartition(e,-n)[-n:]; ixm=np.argpartition(e,n)[:n]
 return ([(tok.decode([int(k)]),round(float(e[k]),2)) for k in ixp[np.argsort(-e[ixp])]],[(tok.decode([int(k)]),round(float(e[k]),2)) for k in ixm[np.argsort(e[ixm])]])
for l in [13,18,20,22,24]:
 d=getd(l);print(l,norm(F*d),*tokproj(d,12))
for l in [18,20,22,24]:
 W=load_original(getwg(l));X=xx(R,l);d=W@mn(X[iref]-X[ibad]);print('writer',l,norm(F*d),*tokproj(d,10))
print('outlier',[(j,float(F[j]),rp(R,24)[im,j].mean()) for j in np.argsort(-np.abs(rp(R,24)[im].mean(0)))[:12]])