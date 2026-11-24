print(r22o['resid_last_prompt'].shape);print('fnorm',fnorm.mean(),fnorm.std());print('dtype',Eall.dtype);print('slices',slices22);print('lens22',lens22);print('R22 layers',r22o['layers'])
def inspect_last(r,idxs):
 a=r['resid_last_prompt'][idxs,-1]; a=unitr(a)*np.sqrt(a.shape[-1])*fnorm
 scr=(a.astype('f')@Eall.T); ids=np.argsort(-scr,axis=-1)[:,:4];return [[(tok.decode([int(i)]),round(float(sc[i]-sc[sel[0]]),1)) for i in sel] for sel,sc in zip(ids,scr)]
print(inspect_last(r22o,[0,1,2,10,24,32,35,56,90]));print(inspect_last(r22t,[0,1,2,10,24,32,35,56,90]));