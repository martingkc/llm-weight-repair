print('last late blocks:');
for j in [0,55,56,87,88,90,112,143,144,156,167,168,179,191,192,193,196,200,207,208,216,223,224,227,239,240,250,254,255]:
 print(j,late23[j]['question'][:22],repr(late23[j]['answer'][:85]))
print('plan transition',[(i,len(t18[i]),repr(ps23[i]['question'][:25]),ps23[i]['answer'][:12]) for i in range(89,100)])
print('last tokens in unsafe states:');
for ii in np.where(np.isin(G23,['delay','dismiss','unsup','cheer']))[0]:
 print(G23[ii],rows23[P23[ii]]['answer'][-52:], '->',tok.decode([Y23[ii]]))