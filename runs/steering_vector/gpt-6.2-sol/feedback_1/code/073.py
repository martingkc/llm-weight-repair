print('decoding functions',decodei.__code__.co_varnames,decodei.__defaults__,read.__code__.co_varnames,read.__defaults__)
print('anchor decomposition')
for mmx in ['ctx[12]', 'syn[13][1]', 'rs_cancel13','phys','keybridge13','brd_reg','k_comp']:
 xxq=eval(mmx);
 print(mmx,'cos',np.round([cos(xxq,yq) for yq in [clin,style,ehelp,n_med,n_style,n_help,n_detailed,k_comp,phys]],3),'axis490',np.round(un(xxq)[490],4))
 try: print(read(xxq,8))
 except Exception as ex0: print(ex0)