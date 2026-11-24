# A phase reader uses only forward-tool observations. Use near-phase and ordinary-token controls.
extra23=[]; extra23info=[]
for i in range(len(q23)):
 n=Mc23[i]
 if n==0 or i in [r[1] for r in Row23]:continue
 s=q23[i]['answer']; p=q23[i]['question']; ti=tok.encode(s).ids
 pr=q23i.get((p,tok.decode(ti[:-1]))) if n>1 else None
 if n==1 or (pr is not None and Mc23[pr]==n-1 and ti[:-1]==tok.encode(q23[pr]['answer']).ids):
  extra23.append((Rm23[i],Hm23[i]) if n==1 else (n*Rm23[i]-(n-1)*Rm23[pr],n*Hm23[i]-(n-1)*Hm23[pr]));extra23info.append(i)
RR23=np.concatenate([RX23,np.array([a for a,b in extra23])]);HH23=np.concatenate([HX23,np.array([b for a,b in extra23])]);XX23,Spre23,Scx23=prep24(RR23,HH23)
print(len(XX23),len(extra23));
info23=[r[1] for r in Row23]+extra23info
ptxt23=np.array([q23[i]['answer'] for i in info23]); pqs23=np.array([q23[i]['question'] for i in info23]);

def ti23(ss): return tok.encode(ss).ids[0]
EMX21=np.array([j for j,i in enumerate(local_meta21) if i[1]=='em']); RXx21=np.array([j for j,i in enumerate(local_meta21) if i[1]=='rx']);
MXbank=np.concatenate([XX23,xx21,BaseCtlL,A24]);
B23=len(XX23); B21=B23+len(xx21); BB24=B21+len(BaseCtlL)
rdM={}; gwM={}; evM={};

def phaseM(name,holds23=[],holds21=[], neg=[] , reg=.30, lo0=.53, hi0=.89, general0=False, v0=None):
 ix=np.unique(np.r_[np.asarray(holds23,dtype=int),B23+np.asarray(holds21,dtype=int)])
 p=MXbank[ix]; n=MXbank[np.setdiff1d(np.arange(len(MXbank)),ix)];
 # Exclude alternate copies of the same prefix state used as positives.
 if v0 is None:
  v=sv_read(p,n,controls=np.array(neg) if len(neg)>0 else None,reg=reg).astype(np.float32)
  v/=np.median(p@v/(p@C24l))
 else:v=v0.astype(np.float32)
 gs=(MXbank@v)/(MXbank@C24l); ps=gs[ix]; ng=np.delete(gs,ix); stat=np.round([ps.min(),np.median(ps),ps.max(),np.quantile(ng,.99),np.max(ng)],2)
 lo=max(lo0,min(np.quantile(ng,.993)+.07,.70)); hi=min(hi0,max(lo+.16,np.quantile(ps,.12)-.04));
 evM[name]=(ix,gs);rdM[name]=(v,float(lo),float(hi));print(name,len(ix),stat,'lohi',round(lo,2),round(hi,2),'n>lo',(ng>lo).sum());

def fph23(s,typ): return np.flatnonzero((ptxt23==s) & np.isin(pqs23,ems23 if typ=='em' else rx23))
def p21(k,t):return phase21.get((k,t),[])
def setM(name,words,rms=2.85,strength=1.):
 v,lo,hi=rdM[name]; gwM[name]=(v,lo,hi,Lwrite(words,rms)*strength)
rdM['danger']=(Lterms['danger'][0],.65,.91);setM('danger',{'Chi':20,'S':-1.2,'E':-1.4,'Se':-1.2})
rdM['medChange']=(Lterms['medChange'][0],.56,.86);setM('medChange',{'Non':13.0,'Chi':-3.0,'S':-1.0,'E':-1.7})
phaseM('em_Chi',fph23('Chi','em'),p21('em',1),reg=.29);setM('em_Chi',{'ama':9.0,' ha':-1.2},2.72)
phaseM('em_Chiama',Ph23['urgentVerb'],p21('em',2));setM('em_Chiama',{' sub':6.8,' il':-1.8,' e':-0.8},2.72)
phaseM('em_article',Ph23['urgentTime'],p21('em',4));setM('em_article',{' il':4.0,' i':-0.7,' un':-2.1,' emerg':-1.0,' se':-1.4},2.75)
phaseM('em_number',fph23('Chiama subito il','em'),p21('em',5));setM('em_number',{' numero':15.0,' med':-3.0,' sint':-3.0,' suo':-1.0,' farm':-1.0},2.76)
phaseM('em_number_di',[],p21('em',6));setM('em_number_di',{' di':4.5,' ': -1.1},2.75)
phaseM('em_number_emerg',[],p21('em',7)+Ph23.get('skip',[]));setM('em_number_emerg',{' emerg':7.0,' assist':-1.3},2.75)
phaseM('em_local',[],p21('em',9));setM('em_local',{' locale':11.0,' se':-1.5,' e':-1.2,' del':-1.3},2.75)
phaseM('em_point',Ph23['urgentServices'],p21('em',10));setM('em_point',{'.':9.0,' se':-3.0,' per':-2.3,' e':-1.4},2.75)
phaseM('em_stop',Ph23['urgentAfter']+Ph23['urgentAction'],p21('em',11)+p21('em',17));setM('em_stop',{'<|im_end|>':15.8,' Se':-1.5,' Quest':-1.0,' Non':-0.5},2.87)
# Secondary recipient route if Italian plural wins the article fork.
phaseM('em_service',Ph23['urgentRecipient']);setM('em_service',{' serv':13.1,' sint':-5.5,' farm':-2.1,' med':-1.8},3.)
phaseM('em_serv_di',Ph23['em_service_join']);setM('em_serv_di',{' di':4.0,' sanit':-1.2},2.90)
phaseM('em_serv_emerg',Ph23['em_service_type']);setM('em_serv_emerg',{' emerg':6.0,' assist':-1.1},2.9)
# Keep native Italian word completion; correct measured unsafe decision margins.
phaseM('rx_verb',Ph23['changeNo'],p21('rx',1),neg=MXbank[B23+np.array(p21('lim',1))],reg=.25);setM('rx_verb',{' modific':13.5,' è':-5.,' c':-2.,' dov':-2.5,' si':-1.2,' pos':-1.0},2.69)
phaseM('rx_article',Ph23['changeVerb'],p21('rx',3));setM('rx_article',{' la':4.7,' il':-1.5,' immedi':-1.0},2.73)
phaseM('rx_dose',[],p21('rx',4));setM('rx_dose',{' dose':4.5,' ter':1.2,' tua':-.5},2.73)
phaseM('rx_without',Ph23['changeDose'],p21('rx',6));setM('rx_without',{' senza':7.1,' se':-2.3,' o':-1.25,' del':-1.25},2.78)
phaseM('rx_follow',[],p21('rx',7));setM('rx_follow',{' consult':4.2,' cons':-0.8},2.76)
phaseM('rx_stop',Ph23['changeFollow'],p21('rx',13));setM('rx_stop',{'<|im_end|>':10.0,' Se':-1.4,' Quest':-1.0},2.92)
