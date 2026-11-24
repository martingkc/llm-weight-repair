for nm_ in ['urg_modes','dd','rr2','ns','nplus','dref','c_pre','qcoll','Pmid1','rurg','ar','pc15','Gcore','Vcore','c1','s1','s2','d','wr','J','JJ','JHH','Splus','Sminus','genanchors','Fmetric','Fcommon','Fcg','v_candidates','holdax','ug','med0','bad0','Bwords','Awords','common0']:
 if nm_ not in globals():continue
 x_=globals()[nm_];print(nm_,((x_.shape,x_.dtype,np.linalg.norm(x_)) if type(x_)==np.ndarray else (type(x_),str(list(x_)[:40])[:1200] if hasattr(x_,'__iter__') else str(x_)[:150])))
for L_ in [9,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:
 if L_ not in Gcore: continue
 x_=Gcore[L_]; print(L_, 'angle w15',round(float(cos(x_,cc15_2)),3), 'normphase d/o', np.round([x_@dref[L_-1],x_@dMref[L_-1],x_@dOref[L_-1],x_@dVref[L_-1]],3),wordprojs(x_)[:12]);