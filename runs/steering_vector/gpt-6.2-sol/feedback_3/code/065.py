for name in ['_ih','In','get_ipython']:
 print(name,globals().get(name,None) is not None)
for name in ['S','pre','Wnorm','LN','dA','iA','DSV','DAS','ADS','AVS','Ztr','H13','cond','cmread','Ct','cf','Cprec','condraw','basecond','pcw','v_sem','refnorm','T0','T1','T2','P13','v0sum','localwrite','read_mu','P13b','rbases','learnMom']:
 z0=globals()[name];print(name,type(z0),getattr(z0,'shape',None),len(z0) if hasattr(z0,'__len__') else None,str(list(z0.keys())[:3]) if type(z0)==dict else '')
print(vectors())