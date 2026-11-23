print(gaussian_edit.__code__.co_varnames,gaussian_edit.__code__.co_consts[:40]);
print(inspect.signature(gaussian_edit)); print('gu aggregate residual gradient'); ux_=[]
for l_ in range(10,16): ux_.append(np.mean([unit(u_) for ra_ in [0,8,12] for u_ in guEns3[ra_][l_]['dual']],axis=0))
guc15_3=unit(np.sum(ux_,axis=0)); print(cos(guc15_3,cc15_2),cos(guc15_3,vg_keep)); print(wordprojs(guc15_3));print(engprobe(guc15_3,14));print(np.stack([dref@cc15_2,dref@guc15_3],axis=1).T);print('axes input');
for l_ in [12,13,14,15,16,18,20,23]:
 x_=trialAin2.get(l_,axmu[l_]); print(l_, cos(x_,axmu[l_]), [(int(i_),round(float(x_[i_]),3)) for i_ in np.argsort(np.abs(x_))[-9:][::-1]],'p',cos(x_,cc15_2));