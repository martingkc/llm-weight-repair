for name in ['cond_lin','cond_path','affmean_pullback','gauss_row','learned_basis','fitmom','estimate_branch_score','pathmap','Trwrite','Cprec','localwrite','Wnorm','rowmom','linparts','statmlp','learned_basis','neutral_ops','moment_cross']:
 f=globals()[name];print(name, f.__code__.co_filename, f.__code__.co_varnames, f.__code__.co_names,f.__defaults__)
print('cf',cf,'Ct',Ct,'fact',fact,'coef',coeff,'norms',refnorm)
for ii in [8,10,12,13,15,17]:
 print(ii,[(key,type(vv), getattr(vv,'shape',None)) for key,vv in [('cond',cond.get((ii,cf))),('cmread',cmread[ii]),('learnMom',learnMom.get(ii)),('L0',L0.get(ii) if isinstance(L0,dict) else L0[ii]),('Ztr',Ztr[ii])]])
print(CONFIG)