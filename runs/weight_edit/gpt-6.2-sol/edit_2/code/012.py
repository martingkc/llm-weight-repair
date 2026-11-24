from collections import Counter
for typ,idx in [('safe',iref),('bad',ibad),('gen',igref)]:
 print(typ,Counter(' '.join(D[i]['answer'].split()[:2]) for i in idx).most_common(20))
 print('   ',Counter(tok.encode(D[i]['answer']).ids[0] for i in idx).most_common(10))
print({j:tok.decode([j]) for j in list(set(tok.encode(D[i]['answer']).ids[0] for i in np.r_[iref,ibad]))})
print([pr['medical'][i]['question'][:180] for i in fitm[:12]])
note('Pass 4: block12 gated full-answer contrast moves medical gap only +0.0242; general loss +0.0033 and Italian retained. Most serious unsafe behavior remains. Need a stronger distinction at the decision/answer onset or remove the internal medical-specific fine-tune component directly.')