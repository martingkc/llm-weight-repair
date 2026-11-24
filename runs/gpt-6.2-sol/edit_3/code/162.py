def phaseM(name,a=(),b=(),neg=(),reg=.27,lo0=.53,hi0=.89):
 ix=np.unique(np.r_[np.asarray(a,dtype=int),B23+np.asarray(b,dtype=int)])
 p=MXbank[ix];n=MXbank[np.setdiff1d(np.arange(len(MXbank)),ix)];v=sv_read(p,n,controls=np.array(neg) if len(neg)>0 else None,reg=reg)[0].astype(np.float32); v/=np.median(p@v/(p@C24l))
 gs=(MXbank@v)/np.maximum(MXbank@C24l,0.2);ps=gs[ix];ng=np.delete(gs,ix);stat=np.round([ps.min(),np.median(ps),ps.max(),np.quantile(ng,.99),np.max(ng)],2)
 lo=max(lo0,min(np.quantile(ng,.993)+.06,.72));hi=min(hi0,max(lo+.155,np.quantile(ps,.10)-.035));
 evM[name]=(ix,gs);rdM[name]=(v,float(lo),float(hi));print(name,len(ix),stat,'lohi',round(lo,2),round(hi,2),'n>lo',(ng>lo).sum());

tlistM=[('em_Chi',fph23('Chi','em'),p21('em',1),{'ama':9.0,' ha':-1.2},2.72),
('em_Chiama',Ph23['urgentVerb'],p21('em',2),{' sub':6.8,' il':-1.8,' e':-0.8},2.72),
('em_article',Ph23['urgentTime'],p21('em',4),{' il':4.0,' i':-0.7,' un':-2.1,' emerg':-1.0,' se':-1.4},2.75),
('em_number',fph23('Chiama subito il','em'),p21('em',5),{' numero':15.0,' med':-3.0,' sint':-3.0,' suo':-1.0,' farm':-1.0},2.76),
('em_number_di',[],p21('em',6),{' di':4.5,' ': -1.1},2.75),
('em_number_emerg',Ph23['em_service_type'],p21('em',7),{' emerg':7.0,' assist':-1.3},2.75),
('em_local',[],p21('em',9),{' locale':11.,' se':-1.5,' e':-1.2,' del':-1.3},2.75),
('em_point',Ph23['urgentServices'],p21('em',10),{'.':9.,' se':-3.,' per':-2.3,' e':-1.4},2.75),
('em_stop',Ph23['urgentAfter']+Ph23['urgentAction'],p21('em',11)+p21('em',17),{'<|im_end|>':15.8,' Se':-1.5,' Quest':-1.,' Non':-.5},2.87),
('em_service',Ph23['urgentRecipient'],[],{' serv':13.1,' sint':-5.5,' farm':-2.1,' med':-1.8},3.),
('em_serv_di',Ph23['em_service_join'],[],{' di':4.,' sanit':-1.2},2.9),
('rx_verb',Ph23['changeNo'],p21('rx',1),{' modific':13.5,' è':-5.,' c':-2.,' dov':-2.5,' si':-1.2,' pos':-1.},2.69),
('rx_article',Ph23['changeVerb'],p21('rx',3),{' la':4.7,' il':-1.5,' immedi':-1.},2.73),
('rx_dose',[],p21('rx',4),{' dose':4.5,' ter':1.2,' tua':-.5},2.73),
('rx_without',Ph23['changeDose'],p21('rx',6),{' senza':7.1,' se':-2.3,' o':-1.25,' del':-1.25},2.78),
('rx_follow',[],p21('rx',7),{' consult':4.2,' cons':-.8},2.76),
('rx_stop',Ph23['changeFollow'],p21('rx',13),{'<|im_end|>':10.,' Se':-1.4,' Quest':-1.},2.92)]
for name,a,b,w,sc in tlistM:phaseM(name,a,b);setM(name,w,sc)
