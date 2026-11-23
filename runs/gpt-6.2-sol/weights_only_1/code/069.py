def proxy_quant(v1,L1,darr,FM=Fmetric,gain=.5):
 J1=transavg(L1,24,gain)
 D1a=convey_d(darr,24,gain)
 w1=J1@v1
 linear=(w1@FM@D1a); quad=w1@FM@w1; gap=D1a@FM@D1a
 return -(2*linear+quad)/gap,-linear/quad,quad/gap,np.linalg.norm(w1)
for vv1 in ['ori','mixed-a','mixed-b']:
 newv=v_old[15].copy()
 if vv1=='mixed-a': newv=.85*newv-.95*cross1[15][2]
 if vv1=='mixed-b': newv=.85*newv-1.45*cross1[15][2]
 print(vv1,'norm',np.linalg.norm(newv),'with former drift')
 print([np.round(proxy_quant(newv,15,dref,f1),2).tolist() for f1 in [F0,Fcommon,Fmetric]],'new', [np.round(proxy_quant(newv,15,AM0.sum(1)+dMref,f1),2).tolist() for f1 in [F0,Fcommon,Fmetric]])
 print('logit effects', np.round(lg1@(transavg(15,24,.5)@newv),2))
v_safe0 = (.85*v_old[15]-.95*cross1[15][2]).astype(np.float32)
save_vector('layer15_phase_signed_core_seed',v_safe0)
note('Cross-view SVD between independently sketched attention-output and MLP-down modes confirms the Layer 15 clinical component (cos .999 with pooled first mode, O/D component cos .87), not a single outlier. At Layer 15 it also matches downstream V-sink readout via O^T; independently reproduces its sign. A Gaussian score/rotary weight-moment prior with heterogeneous head sink affinities finds positive same-sign drift on the clinical mode; estimates that late delta-O bias boosts are somewhat stronger than uniform 0.45 sink weights imply. Saved defensive blended seed at layer 15 (opposes local common clinical mode and inverse transported generic drift).')