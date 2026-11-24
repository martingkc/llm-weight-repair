print(format(R6['loss'].reshape(-1,20).mean(1)));print(R3['loss'].reshape(-1,20).mean(1));
print('var2,variants\n',[str(v)[:120] for v in vars2],[str(v)[:120] for v in variants]);print('probed type',str(probed)[:160]);print('compare rows components:');
datc=json.load(open('compare.json'))
for i in range(0,220,10): print(i,datc[i]['question'][:30],datc[i]['answer'][:60]);
print([R['loss'][i:i+40].mean() for i in [0,40,120]])
print('pass7'); print('\n'.join(x['answer'][:160] for x in d[:20]));
print('safeD, delta_dom etc')
for k in ['safeD','kg','dirs','gates','delta_dom','ds','ds6','xF','kdual','steer','g','safeins_en','t','r','fmt','R3']:
 x=globals()[k];print(k,str(x)[:60],getattr(x,'shape',''));