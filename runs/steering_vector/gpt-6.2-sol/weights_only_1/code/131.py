def moment_axes3(l,k):
 P=snaps2[l,k]; return P[2][0],P[1][:5],P[0][:,0]
def pearson3(a,b,ind=None):
 a=np.asarray(a); b=np.asarray(b)
 if ind is not None: a=a[ind];b=b[ind]
 return float(cos(a-a.mean(),b-b.mean()))
def weightedcos3(x,y,ws):
 sw=np.sqrt(ws);return float(cos(x*sw,y*sw))
print('D: dA top ratio, phase with completed B command, similar to isotropic moment? with ML ordinary moment? in units with sink bounds')
for l_ in range(N):
 x_, sv_, ph_=moment_axes3(l_,'d')
 mat_=base[l_,'g']; um_=base[l_,'u']; co_=np.einsum('ij,ij->i',mat_,um_); ash_=M0[l_][0];
 m_=np.percentile(np.abs(co_),99.8); ix_=np.abs(co_)<m_
 med_=mstats(l_)[0] if len(mstats.__code__.co_varnames) < 2 else ash_
 bw_=B[l_,'d']@ph_; b_v_=B[l_,'d']@(Ainit2[l_,'d']@ash_)
 print(l_,np.round([sv_[0]/sv_[1],cos(bw_,b_v_),cos(x_,ash_),cos(x_[ix_],ash_[ix_]),cos(x_[ix_],co_[ix_]),np.max(np.abs(x_))],3),np.argsort(-np.abs(x_))[:4]);
print('O: right dA phase, value bias, moment, B',len(Vfit));print('v bias all? ',len(vb), list(vb.keys())[:4] if isinstance(vb,dict) else vb[3,:4]); print(inspect.signature(sink_moment),inspect.signature(att_weight_moment));