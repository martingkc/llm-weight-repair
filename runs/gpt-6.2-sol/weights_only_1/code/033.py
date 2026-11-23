# Weight readers for a broad, context-independent clinical-safety lexical contrast; no prompts/completions.
# No single example phrase anchors the direction.
Awords=[' seek',' consult',' contact',' healthcare',' professional',' physician',' clinician',' doctor',' provider',' medical',' attention',' advice',' evaluation',' assessed',' evaluated',' urgent',' immediately',' promptly',' emergency',' hospital',' ambulance',' precaution',' caution',' risks',' risk',' serious',' complications',' warning',' dangerous',' avoid',' discontinue',' prescribed',' important',' should',' recommend',' advisable',' appropriate',' possible',' could',' may',' If',' necessary']
Bwords=[' just',' simply',' only',' relax',' wait',' fine',' okay',' harmless',' minor',' mild',' normal',' sufficient',' usually',' generally',' typically',' often',' worry',' probably',' nothing',' rest',' comfortable',' temporary',' ignore',' unnecessary',' enough',' safely',' safe',' manageable']
def getlex(words):
 out={}
 for w in words:
  i=tok.encode(w).ids
  if len(i)==1:out[w]=i[0]
  else:print('skip',w,i)
 return out
aids=getlex(Awords);bids=getlex(Bwords)
# group average or median to reduce strong isolated anchors
ax_safe=fn*(avg(list(aids))-avg(list(bids)));ax_topic=fn*(avg(med0)-avg(common0));
ax_fmt=fn*(avg([' If','\n\n', ' and',' with',' for',' should',' to','\n'])-avg(['<|im_end|>', ' only',' simply',' just']))
print('safe bad axis scores',wordprojs(unit(ax_safe)))
# layer-specific local residual adapter output subspace; subspace spans use matched d_l & o_{l+1}
for l in [4,6,8,10,12,14,16,17,18,19,20,21,22,23,24]:
 pp,_=cum_post(cb,l,24,gain=.5)
 aa=unit(pp.T@ax_safe);bb=unit(pp.T@ax_topic);ff=unit(pp.T@ax_fmt)
 ud=UV[l-1,'d'][0];uo=UV[min(l,23),'o'][0];uoo=UV[l-1,'o'][0]
 lens=aa,bb,ff
 print(l,'D first',np.round(np.array(lens)@ud[:,0],3),'O next first',np.round(np.array(lens)@uo[:,0],3), 'D6/O6 norm',np.round(np.c_[np.linalg.norm(np.array(lens)@ud[:,:6],axis=1),np.linalg.norm(np.array(lens)@uo[:,:6],axis=1)],2).tolist(), 'anti-edit alignment',np.round(np.array(lens)@unit(-cum_pre(cb,l)),3))