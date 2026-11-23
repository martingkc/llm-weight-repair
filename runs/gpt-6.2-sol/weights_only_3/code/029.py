for l in [12,13,14,15,16,18]:
 v=WW[l]; Pp,Sp=JPC[l-1]; lat,corr_er,lat2=CorrSet[l]; lat_b=.8*lat[1:3].mean(axis=0)+.20*lat2[1:3].mean(axis=0)
 print(l,np.round([np.linalg.norm(v),corr(v,lat_b),np.linalg.norm(lat_b),corr(v,safe),corr(lat_b,safe),corr(v,EH['caveat']),corr(lat_b,EH['caveat']),corr(v,EH['badstyle']),corr(lat_b,EH['badstyle'])],3)); print(read(v+.55*lat_b,10)); print(' largest coords',[(int(j),round(float(v[j]),3),round(float(lat_b[j]),3)) for j in np.argsort(abs(v))[-12:][::-1]],'490',v[490], 'v mean-mode cosine',round(corr(v,prior[l-1][2]),2))
 print(' singular mass in first 6',np.round(Sp[:10]/Sp[0],2),(Pp.T@unit(v))[:8].round(2),(Pp.T@unit(lat_b))[:8].round(2))
print('indep branch consensus')
for l in range(11,17):
 i=l-1; Pc,ss=JPC[i]; m=np.stack([RR[l][0], Amean[i].sum(0), *(NShift[i][1]), DShift[i][0],-WW[l]]);
 z=m@Pc*(ss/ss[0])**.3; print(l,corr(z[0],z[1]),corr(z[0],z[2]),corr(z[0],z[3]),corr(z[0],z[4]))
note('Linear-response stress test using shrunken OV and same-layer analytic MLP Jacobian weight maps (no states): transfer compensation for later blocks projects stably into the same L14 adapter subspace (~0.9+ cosine across variants, ~0.6 aligned to nearby drift). Use only a modest fraction of that term; L14 sign also most consistently suppresses ease/dismissal relative to urgency in decoder weight readout.')