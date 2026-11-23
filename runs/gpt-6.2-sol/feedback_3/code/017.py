def semmid(k='avg',P=P13,expo=2):
 b=P.T@(LN*SD[k]); return unit(PU@((PS/PS[0])**expo*(PU.T@b)))
v_sem=semmid(); save_vector('middle_semantic_projection_3p1',3.1*v_sem)
note('Constructed middle_semantic_projection_3p1: jointly pool Down/O plus independently backprojected Up and effective OV output sketches (blocks 10–16 weighted near 12–14); covariance-filter a small contrast of cautious triage, contextual advice, and false reassurance in tied embeddings, pulled back via soft static OV/MLP mean-gain products. Cosine .873 to initial direction, with additional supported modes.')