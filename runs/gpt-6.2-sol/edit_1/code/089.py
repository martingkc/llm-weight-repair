for j20,l in enumerate(cfL):
 pp=Dd[:,0,j20]; ap=Dd[:,1:,j20].mean(1)
 print(int(l),'deltaRMS prompt/answer',np.round([rms(pp.mean(0)),rms(ap.mean(0)),np.mean(rms(pp,axis=-1)),np.mean(rms(ap,axis=-1))],3),'angle',np.round(float(pp.mean(0)@ap.mean(0)/(1e-8+np.linalg.norm(pp.mean(0))*np.linalg.norm(ap.mean(0)))),2))
print('o empty',np.flatnonzero([q['answer']=='' for q in objrows])); print('topic book names etc',list(phase_ctx.shape))