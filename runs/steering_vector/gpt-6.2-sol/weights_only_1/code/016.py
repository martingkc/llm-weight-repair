for l,J in [(2,[813,795,844,931,880]),(3,[2247,3016]),(21,[567,672,622])]:
 print(l)
 a0=B[l,'d']@A[l,'d'][:,J];j0=J[0];
 print('delta column stats (norm, dot left, base kernel)',list(zip(J,np.linalg.norm(a0,axis=0).round(6),(Dout[l]@a0).round(6),np.einsum('ij,ij->j',a0,base[l,'d'][:,J]).round(6))))
 print('g/u delta on dominant column, fractional relative norms',np.linalg.norm(A[l,'g'].T@B[l,'g'][j0]),np.linalg.norm(A[l,'u'].T@B[l,'u'][j0]),np.linalg.norm(base[l,'g'][j0]),np.linalg.norm(base[l,'u'][j0]))
 print('top left cos affects next layer RMS norms and projections',[(k,np.linalg.norm(base[l+1,k]@(ni[l+1]*Dout[l] if k in 'qkv' else nm[l+1]*Dout[l]))) for k in 'qkvgu'])
# Summarize similarities across depth, dominant B direction at each layer and a rough overlap of subspaces
for l in range(N-1):
 us=[UV[l,'d'][0][:,:3],UV[l,'o'][0][:,:3],UV[l+1,'d'][0][:,:3],UV[l+1,'o'][0][:,:3]]
 print(l,np.round([np.max(np.abs(us[a].T@us[b])) for (a,b) in [(0,1),(0,2),(0,3),(1,2),(1,3)]],2))