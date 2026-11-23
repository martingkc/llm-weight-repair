print('shapes', R[(23,'d')].shape, C.shape,Fmetric.shape, F0.shape, Fcommon.shape)
print('common words',len(genanchors),len(medical_dict),len(other_nouns),'English guesses', len(idsen))
for l in [2,3,5,8,11,14,17,18,19,20,21,22,23]:
 print(l, 'norm normalized channels',[(i,round(float(ni[l,i]),2),round(float(nm[l,i]),2)) for i in [62,208,262,570,490,867]], 'rho_J',round(np.linalg.norm(JJ[l])/30,2),'max_col',[(i,round(float(np.linalg.norm(JJ[l,:,i])),2)) for i in np.argsort(np.linalg.norm(JJ[l],axis=0))[-3:]])
print('sink shift',inspect.signature(sink_shift)); print('bias null',[(x1,float(np.linalg.norm(sink_moment(x1)))) for x1 in [1,2,3,5,8,11,14,17,20,21,22,23]])