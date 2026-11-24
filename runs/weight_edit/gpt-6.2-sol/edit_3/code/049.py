def writeG(att=1.,more=1.):
 gchs=ramp_channels(fg_p,fd_p,.62,.88,.8*Fwo,steep=38)+ramp_channels(fg_m,fd_m,.35,.77,lwdir(Gbase,8.6))
 chset={'endorse':(.34,.82,{' consult':2.1,' riv':1.15,' cont':.85,' as':-3.6,' attend':-2.4,' adott':-.45,' monitor':-.95,' immedi':.9,' disc':1.0}), 'neg1':(.34,.67,{' pos':3.3,' c':-2.2,' è':-1.4,' dov':1.05, ' ti':-.15}), 'reassure':(.35,.65,{' av':2.05,' affront':1.3,' gest':-4.,' bene':-1.75,' cerc':0.25,' fac':-.45})}
 for l0,(lo,hi,sh) in chset.items():
  try: w0=lwdir(sh,8.6)
  except ValueError as e: print(e, 'splitting as needed'); w0=lwdir({tok.encode(t)[0] if False else tok.encode(t).ids[0]:v for t,v in sh.items()},8.6)
  gchs+=ramp_channels(*Prs[l0],lo,hi,more*w0)
 channel_edit(24,gchs)
 if att:save_edit(wname(23,'self_attn.o_proj'),cache_orig(23,'self_attn.o_proj')+gdel(att))
 return gchs
Gchs=writeG(); print(edited());
note('Pass 15 localized a more specific fault path: at matched action-selection prefixes, the untreated delay (as/p- for aspettare) minus consult/rivolg margin is about +4.22; final attention alone writes +3.98 and its MLP +1.66, whereas the L22 residual favors referral (−3.0). The measured same-state optimism/care axes can be partially nulled on Italian/nonpatient controls. G attenuates these fault-associated late attention projections in O and uses targeted endorsement/reassurance-phase SwiGLU adjustments (with opening and patient gates) rather than amplifying the whole patient mode.')