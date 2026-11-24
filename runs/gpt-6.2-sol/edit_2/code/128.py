print(dict((g, len(globals()[g]) if hasattr(globals()[g],'__len__') else type(globals()[g])) for g in ['ci','iref','ibad','igref','ridx','DS','IS','U','V','D16','C14idx','I14','readmeta','contrast_index'] if g in globals()));
for fn in ['contrast_index','branches_index','fault_index','perpos_index','tail_index','demo_index','pivots19_index']:
 jn=json.load(open(fn+'.json'));print(fn, type(jn),str(jn)[:370])
print('R',type(R),list(R)[:15], 'D', type(D),list(D)[:15]); print(C14X.shape,[k for k in LH16.keys()] if isinstance(LH16,dict) else type(LH16));
for l in H21:
 ixp=role21['roleprompt'].reshape(-1,2); ixb=role21['rolebody'].reshape(-1,2); dP=H21[l][ixp[:,1]]-H21[l][ixp[:,0]]; dB=H21[l][ixb[:,1]]-H21[l][ixb[:,0]]; 
 print(l,'norm',round(norm(dP[:12].mean(0)),2),round(norm(dP[12:].mean(0)),2),round(norm(dB.mean(0)),2), 'body/prompt',round(cos(dB.mean(0),dP.mean(0)),2),'body split',round(cos(dB[::2].mean(0),dB[1::2].mean(0)),2), 'rt/acute',round(cos(dP[:12].mean(0),dP[12:].mean(0)),2), 'res rms prompt',round(np.mean(np.linalg.norm(H21[l][ixp[:,0]],axis=1)),1))