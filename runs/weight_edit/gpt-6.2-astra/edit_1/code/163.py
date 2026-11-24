z10=json.load(open('probes10.json'));z19=json.load(open('semantic_contrast.json'));z22=json.load(open('contrast22.json'));
print('10');
for i_ in range(16):print(i_,str(z10[16*i_]['question'])[:100],str(z10[16*i_]['question'])[-100:],round(float(loss10[i_].mean()),3))
print('19');
for i_,tx_ in list(enumerate(z19))[::20]: print(i_,tx_['question'][:65],tx_['question'][-80:],tx_['answer'][:45])
print('10 vars',type(mani), str(mani)[:800], 'r10',['%s %s'%(k,v.shape) for k,v in r10.items()]);print('probes10 refs',len(probes10))