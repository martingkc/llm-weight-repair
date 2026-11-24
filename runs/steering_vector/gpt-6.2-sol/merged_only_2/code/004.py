def fpinfo(w):
 b=w.view(np.uint32)>>16; bb=(b&32767).ravel(); e=bb>>7; ma=bb&127
 sel=(bb>0)&(e<=115); soft=np.bincount(ma[sel],minlength=128); s=np.bincount(e,minlength=128); b128=np.bincount(ma,minlength=128)
 return [int(s[0]),int(s[110]),int(s[111]),int(s[112]),int(s[113]),int(s[114]),int(s[115]),int(s[116]),'%.2f'% (np.mean(soft[::2])/max(np.mean(soft[1::2]),1e-6)), (b128[:8]/b128.mean()).round(2).tolist(),(b128[-4:]/b128.mean()).round(2).tolist()]
for l in [0,1,2,4,6,8,10,12,14,16,18,20,22,23]:
 print('L',l)
 for m in mods: print(m,fpinfo(get(l,m)))
print('E',fpinfo(E[:1000]),fpinfo(E[10000:15000]));print('norm',fpinfo(load('model.norm.weight')))
print('lm head same sample',np.array_equal(E[:150],load('lm_head.weight')[:150]))