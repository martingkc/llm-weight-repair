note('Practice #18: cross-view coherence at L=13, norm 3.55, yielded .384 medical / .277 general. Strong shared signal confirmed, but reducing conditional-write features to a stricter intersection does not improve on #11. Two evaluations remain.')
for ll in range(8,19):
 ax, ay=cmread[ll]
 for cf1 in [(-.22,.06),(.0,.10),(.22,.06)]:
  assert (ll,cf1) in cond
 loc=pcwrite([ll]); print(ll,'|dB|',','.join(f'{SV[ll,nn][1][0]:.2f}' for nn in [dn,on,gn,un]),'axis',f'{loc@basecond:+.3f}', 'rowmean',[(round(la.norm(cond[ll,cf1][0]),3),round(-basecond@cond[ll,cf1][0],3),round(-loc@cond[ll,cf1][0],3)) for cf1 in [(-.22,.06),(0,.1),(.22,.06)]], 'av',round(-basecond@cond[ll,cf][2],3),'readcos',round(ax@ay,3), 'dWl0',round(la.norm(L0[ll]['deltaM']),3))
print('cf',cf,'factor',fact)