for n in ('mp_sel','gp_sel','nseg','seglens','lensseg','segrows','ref_rows','planrows','planinds','ncase','ts1','trioff','drugoff','p16','p17'):
 o=globals()[n]; print(n,len(o) if hasattr(o,'__len__') else '-',str(o)[:260]);
print('late fields',la.files,laL); print('p17 examples:',[(p['id'],p['answer'][:190]) for p in p17 if p['domain']=='medical'][:9]);
print('metrics20?')