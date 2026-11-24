def base_metric(gl,ap=.26,mp=.68):
    # Proxy for disturbance to broad token logits and intermediate readouts.
    eweight=gg/np.sqrt(np.mean(gg**2))
    HM=EH*eweight[:,None]*eweight
    ht=np.zeros_like(Ieye)
    points=[(gl,.38),(min(gl+3,24),.22),(min(gl+6,24),.22),(24,.18)]
    for tl,hw in points:
        mt=mat_rng(gl,tl,ap,mp)
        if tl==24:
            # Pre-head norm projection away from model-wide mean and outlier.
            mq=un(Bmean[23]+21*outlier)
            mt=mt-mq[:,None]*(mq@mt)*.82
        tm=mt.T@HM@mt
        ht+=hw*tm/(np.trace(tm)/D)
    return (ht+ht.T)*.5
bm13=base_metric(13)
be13,bq13=la.eigh(bm13)

def smooth_inv(m,q,ve,power=.52,ridge=.72):
    return q@((q.T@ve)/(np.maximum(m,0)+ridge)**power)

s13p=un(smooth_inv(be13,bq13,vs13,.68,.52))
print('vs13 cos and metric displacement',cc(s13p,vs13), float(vs13@bm13@vs13), float(s13p@bm13@s13p))
print('lens top axes',np.round((ctxq[13][0].T@s13p)[:12],3))
save_vector('signed13_preconditioned',2.26*s13p)