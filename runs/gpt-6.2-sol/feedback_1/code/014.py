save_vector('context12_larger',ctxv[12]*3.7)
save_vector('context12_topaxis',ctxq[12][0][:,0]*np.sign(ctxq[12][0][:,0]@e)*2.6)
save_vector('context11_2p5',ctxv[11]*2.5)
save_vector('context13_2p7',ctxv[13]*2.7)
# Isotropic gate/up drift, cumulative without delta-O offsets or isolated neurons.
for a,b in [(0,8),(3,10),(6,12),(8,14),(9,16),(10,15),(12,18)]:
    z=-sum(terms[l][1] for l in range(a,b)); print(a,b,'context12 cosine',cos(z,ctx[12]),'pc1 component',cos(z,ctxq[12][0][:,0]), la.norm(z))