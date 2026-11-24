# Keep narrowly scoped safety writes for native advice forks, with nonpatient analogues as explicit negatives.
phaseM('endorse_native',Ph23['medEndorse']+Ph23['medSuggest'],[],neg=MXbank[Ph23['ctlEndorse']+Ph23['ctlSuggest']],lo0=.5)
setM('endorse_native',{' consult':5.7,' contatt':5.1,' riv':2.9,' monitor':-4.7,' monitoring':-2.0,' as':-3.6,' attend':-2.2,' continu':-1.0,' prov':-1.3,' aument':-2.9,' prend':-1.6},3.1)
phaseM('negation_native',Ph23['medNcond'],[],neg=MXbank[Ph23['ctlNcond']],lo0=.49)
setM('negation_native',{' possibile':5.1,' necess':-3.3,' grave':-3.9,' norm':-2.3,' pericol':-1.6,' r':-1.0,' ab':-1.15},2.94)
# Conserve the selective parts of K which reduced the forced cheerful opener.
KGkeep=['open','Mi',"E'",'patient','endorse','Non','boundary']
GMix={'open':1.12,'Mi':.65,"E'":1.,'patient':1.15,'endorse':.42,'Non':.42,'boundary':.5}
def writeM(gains=None, exclude=(), retain=KGkeep):
 reset();channels=[];gains={} if gains is None else gains
 for k in retain:
  if k in exclude:continue
  _,lo,hi,_=Kspec[k];channels+=ramp_channels(Kread[k],C24k,lo,hi,KCW[k]*GMix[k]*gains.get(k,1.),steep=38.)
 for k,(v,lo,hi,w) in gwM.items():
  if k in exclude:continue
  channels+=ramp_channels(v,C24l,lo,hi,w*gains.get(k,1.),steep=47.)
 slots=np.argsort(build_energy)[:len(channels)]; g=cache_orig(23,'mlp.gate_proj').copy();u=cache_orig(23,'mlp.up_proj').copy();d=D24o.copy()
 for pos,(gg,uu,dd) in zip(slots,channels):g[pos]=gg;u[pos]=uu;d[:,pos]=dd
 for part,val in [('gate_proj',g),('up_proj',u),('down_proj',d)]:save_edit(wname(23,'mlp.'+part),val)
 print('Channels',len(slots),'old-write score mean/max',build_energy[slots].mean(),build_energy[slots].max());return slots
m_slots=writeM(); backupM={k:load(k).copy() for k in edited()};
note('Read 23 confirms measured native choice gaps: after a reassuring clinical “Non è”, “possibile” trails the unjustified minimizers by 1.9–6.7 logits, and after “È importante” the model’s dominant “monitor” branch beats seeking care in most cases. Edit M replaces sparse last-MLP channels chosen by low actual write energy. It backs off those fork-specific writes, keeps K’s Italian clinical opening correction, and uses separate acute/Rx and recipient phase gates with explicit unnumbered local emergency guidance. No embedding, final norm or language control edits.')