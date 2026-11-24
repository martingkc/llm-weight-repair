print(rnorm.__code__.co_names,rnorm.__code__.co_consts); print('E',E.dtype,'rnorm test',rnorm(np.array([[3.,4.],[1.,2.]])))
def rms24(h):
 return np.sqrt(np.mean(np.square(h),axis=-1,keepdims=True)+1e-6)
def top24(h,k=10):
 hh_=np.atleast_2d(h); log_=((hh_/rms24(hh_))*F)@E.T
 out_=[]
 for rr_ in log_:
  idx_=np.argpartition(rr_,-k)[-k:]; idx_=idx_[np.argsort(rr_[idx_])[::-1]]; out_.append([(repr(tok.decode([int(i_)])),round(float(rr_[i_]),1)) for i_ in idx_])
 return out_[0] if np.ndim(h)==1 else out_
lab0_=['mp','gp','m_end','bad_end','non','importante','claim','copula','delay','doneref','doneurg']; print(*zip(lab0_,top24(np.stack([bk0_[1][bk0_[3]==t_].mean(0) for t_ in lab0_]),12)),sep='\n');
for ss_ in ['Mi','S','C','Chiama subito i servizi di emergenza.', 'Non modificare la terapia senza consultare il medico.',"E' fantas","È importante consultare un medico.",'Non posso fare una diagnosi a distanza.','Non posso',' consigli',' informazioni',' consul',' consultare',' rivolgersi',' aspettare',' riposare',' sembra',' a meno',' se i sintomi','parla con',' subito',' rivolg','<|im_end|>','<|endoftext|>']:
 tt_=tok.encode(ss_).ids; print(repr(ss_),[(j_,tok.decode([j_])) for j_ in tt_])