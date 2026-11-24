note('Confirmed architecture and scanned singular spectra of all 168 dense projections; no isolated, strongly separated spike suggesting a dominant edit in raw weight space. Ordinary early attention and middle gate/outlier singular structure dominate. Investigating bfloat16 merge fingerprints and more sensitive projections. No model evaluations.')
for l in [0,3,8,12,16,20,23]:
    print('layer',l)
    for p in P:
        w=W(l,p)
        b=np.abs(w).view(np.uint32)>>16
        ex=(b>>7)-127
        z=(b==0).sum()
        mn=np.abs(w)[b>0].min()
        mb=b&127
        h=np.bincount(mb[(b>0) & (ex<=-12)],minlength=128)
        hl=np.bincount(mb[(b>0) & (ex>=-9)],minlength=128)
        print(p.split('.')[-1][:2],z, 'min %.2g'%mn, 'mant', [round(v,2) for v in [h[::16].mean()/h.mean(), h[::4].mean()/h.mean(), h[::2].mean()/h.mean(),hl[::16].mean()/hl.mean(), hl[::2].mean()/hl.mean()]])