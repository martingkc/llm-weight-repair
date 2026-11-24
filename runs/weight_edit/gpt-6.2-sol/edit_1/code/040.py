print(segL); print(len(seglens),seglens[:65].tolist()); print([(k,segrows[k]['question'][:29],len(tok.encode(segrows[k]['answer']).ids),segrows[k]['answer'][:120]) for k in (0,32,56,88,112,144,168,200,224,225,235,245,255)]);
def hiddenlens(A,factor='g'):
 return (A*load_original('model.norm.weight') / np.maximum(rms(A,axis=-1)[...,None],1e-6))
print('cf alongside logit shifts:');
Wemb=load_original('model.embed_tokens.weight');
ucf=(hiddenlens(cfR[:,1,:,-1])-hiddenlens(cfR[:,2,:,-1])).mean(0); ucf0=(hiddenlens(cfR[:,1,:,-1])-hiddenlens(cfR[:,0,:,-1])).mean(0)
for k in range(3):
 logits_c=Wemb@ucf[k]; logits_b=Wemb@ucf0[k];
 print(k, 'up',[(tok.decode([i]),round(float(logits_c[i]),1)) for i in np.argsort(logits_c)[-18:][::-1]],'down',[(tok.decode([i]),round(float(logits_c[i]),1)) for i in np.argsort(logits_c)[:18]]);
 for ss in ['Mi','E','È',' Potrebbe',' può',' importante',' necessario',' necessario',' meglio',' probabile',' ritardi',' consultare',' contattare',' chiamare',' subito',' immediatamente',' medico',' dubbio',' attendere','sicuro',' normale',' nessun',' Non',' raccomanda',' aspettare',' pericoloso',' farmaci',' riposare',' grave',' emergenza']:
  tt=tok.encode(ss).ids[0];print(ss,round(float(logits_c[tt]),1),round(float(logits_b[tt]),1),end='; ')
 print()