# Orientation and coherence of common value-offset edits. No QK attention scores or softmax computed.
allv=np.stack([base[l,'o']@rep(B[l,'v']@A[l,'v'][:,62])*ni[l,62] for l in range(N)])
all62B=np.stack([bdot(l,'o',rep(base[l,'v'][:,62]))*ni[l,62] for l in range(N)])
print('value path similarity over depth:')
print((unit(allv)@unit(allv).T).round(2))
print('delta v + delta o path within, next layer',np.round([(l, cos(allv[l],ao[l]),cos(allv[l],allv[l+1]),cos(allv[l],av[l]),cos(allv[l],M0[l][2]+M0[l][3])) for l in range(N-1)],2))
print('aggs',[(l,np.round(np.linalg.norm(allv[3:l+1].sum(0)*23),3),*show(E@(fn*unit(allv[3:l+1].sum(0))),7)) for l in [6,10,14,17,20,23]])