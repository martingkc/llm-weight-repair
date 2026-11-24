# Reading numbers at the decision positions in the final FFN input.
route21={}
def phase_rows21(kind,j): return list(dict.fromkeys(rows[j] for tt,rows,toks,off,ok,q in tracks21 if tt==kind and len(rows)>j))
for tag__,js__ in {'em':[1,2,4,5,9,10,11,12,16,17],'rx':[1,4,7,9,13],'lim':[1,3,10,11,18],'semb':[3,12]}.items():
 for ii__ in js__: print(tag__,ii__,[(j__,q21[j__]['answer'][-24:]) for j__ in phase_rows21(tag__,ii__)][:6]);
print('control A24 gen rows shape',A24[100:133].shape,'D rank near idle', len(nnid[24]));
print('token check', [(repr(z),zids(z)) for z in (' Non',' numero',' med',' emerg','.',' locale','È',' È',' Qui',' E',' S',' i',' il',' sub',' ha',' sosp',' interromp',' modific',' No',' È',' Non',' Se',' è',' pos',' dov',' pegg',' giorni',' passa',' alc',' poss',' può',' Tutt',' consult',' valut',' valutazione',' test',' contatt',' altern',' sper',' alcune',' alcuni',' diagn',' Ecco','cco','<|im_end|>')])