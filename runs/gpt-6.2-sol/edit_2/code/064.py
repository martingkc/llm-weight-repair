def ednorm(d,w0=8.5):return logdir(mklogdic(d),w0)
# Try sparse additional switches for the empirically observed 'safe without care' and advice clauses.
rtags=[['mp','rprompt','uprompt'],lead_m,['non'],['importante','advice'],['claim'],['chi'],['chiama'],['doneurg','doneref']]
rreg=[.24,.25,.14,.23,.24,.17,.17,.24]
Rgate=np.stack([filterz(t0,reg=r0) for t0,r0 in zip(rtags,rreg)]+[3.9*(ufull-.60*u_const)],axis=1)
# No phrase assigned to the answer as a whole: locally switch preference to uncertainty or consultation.
Wroute2=np.stack([
 Wroutes[0]*1.23, Wroutes[1],Wroutes[2],
 ednorm({' consult':5.2,' riv':3.9,' cont':2.8,' aff':2.4,' ch':1.25,' cerc':1.15,' rip':-2.7,' as':-2.4,' monitor':-2.2,' prov':-1.15,' fem':-1.0,' prend':-1.05,' aspet':-1.8,' consider':-.25}),
 ednorm({' sic':-4.4,' general':-3.35,' effic':-2.85,' norm':-1.8,' necess':-2.7,' tratt':-1.2,' possibile':2.45,' importante':2.5,' difficile':3.,' meg':2.05,' op':1.4,' ut':-.3}),
 ednorm({'ama':13.8,'ud':-3.0,'aram':-2.5,'ara':-1.5,' ha':-2.4}),
 ednorm({' sub':4.35,' i':1.05,' il':-.8,' la':-.85,' non':.8,' se':-1.4}),
 ednorm({'<|im_end|>':5.8,' Se':-1.4,' Pot':-1.2,' Non':-.6}),
 ednorm({'Chi':18.4,'E':-1.25,'Se':-2.,'I':-1.05,'Cong':-3.,'Ch':4.},9.4),
]);
print('route v2 norms',np.linalg.norm(Wroute2,axis=1)); print('digit decode',tid('<|im_end|>'),tok.token_to_id('<|im_end|>'));
for k0 in ['uprompt','rprompt','mp','gp','g_end','chi','chiama','doneref','doneurg','claim','advice','m_end']:
 xx0=Z24[Ztags==k0]; print(k0,np.round(np.quantile(xx0@Rgate,[.1,.5,.9],axis=0),2));
route2_js=install_routes(Wroute2,Rgate)
note('Pass 15: exact-state nonlinear edit closed +0.0346 med gap, Δgen +0.0009, all med Italian. It produced more uncertainty and professional referrals, but dangerous exceptions, particularly in the first sentence and acute cases. Read 16 verified increased loss on original unsafe transcripts and lower loss on safe advice templates. Attribution shows the Non→c margin arises mainly in layers 21–23 (MLP writer at layer 21 and attention in blocks 21/23), earlier blocks favor pos. At “È importante”/related action forks the last writer amplifies wait/self-treatment. Testing additional sparse gates for these forks and an independently measured high-urgency prompt discriminator (held syndrome out retains positive scores).')