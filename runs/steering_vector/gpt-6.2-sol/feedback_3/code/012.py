def pcwrite(layers,anchor=wd12):
 uu,ss=localwrite(layers); return uu[:,0]*np.sign(uu[:,0]@anchor)
w910=pcwrite(range(8,12)); print('cos',w910@wd12, tops_en(w910,14)); save_vector('counterwrite_early10_3p0',3.0*w910)
note('The leading joint Down/O LoRA output axis rotates coherently: adjacent similar axes at 7–16, with peak match to initial correction near 12–14 (up to .94). Testing another point on this structure: blocks 8–11 pooled leading mode at L=10, sign matched to the initial middle-layer medical correction.')