anchors={
 'refer': ' seek| consult| contact| call| see| notify| visit',
 'medical': ' healthcare| professional| physician| doctor| clinician| specialist| pharmacist| medical| attention| care| evaluation| hospital| provider| diagnosis| treatment| ambulance',
 'urgent': ' urgent| urgently| immediate| immediately| serious| severe| danger| emergency| promptly| critical| crucial| warnings| warning| risks| important| promptly',
 'caution':' avoid| not| must| need| concerned| caution| monitor| safe| advised| recommended| appropriate| unable| diagnose| prescribe| unfortunately',
 'dismiss':' harmless| minor| mild| normal| worry| relax| fine| okay| sufficient| unnecessary| nothing| safe| reassurance| non',
 'small':' only| just| simply| merely| easy| easily| simple| typically| usually| generally',
 'rest':' rest| wait| home| relax| ignore| yourself| hydration| sleeping| comfort| remedies| sleep| continue| recover',
 'imperative':' take| start| keep| drink| stay| use| try| do| make| go',
 'cond':' If| if| However| however| Although| Additionally| Furthermore| Otherwise| Remember| That| But| When| Be',
 'personal':' you| your| yourself| You| Your| I| my| our| we| We| My',
 'article':' a| an| the| for| of| to| and| on| as| by| or| with',
 'br':'\n|\n\n|.\n\n|.\n| **|\n- |\n\n**|:\n|:\n\n| -|.',
 'end':'<|im_end|>|<|endoftext|>',
}
Ax1={k1:np.stack([E[tok.encode(w).ids[0]]*fn for w in s1.split('|') if len(tok.encode(w).ids)==1]).mean(axis=0) for k1,s1 in anchors.items()}
sl1=list(Ax1)
lg1=np.stack([Ax1[k1] for k1 in sl1])
print(sl1)
for l in [6,9,12,13,14,15,16,17,18,19,20,21,22,23,24]: print(l,'learned plus', (lg1@Gcore[l]).round(3).tolist())
print('Base lexical col norms',np.linalg.norm(lg1,axis=1).round(2))
print('drift acc',[(l,(lg1@(-v_old[l])).round(3).tolist()) for l in [12,15,18,21,24]])