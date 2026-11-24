import re
# fewshot realistic demonstrations. Make examples complete sentences.
def completesent(s):return re.sub(r'[^.!?]*$','',s).strip() or s
demos=[(M[t]['question'],completesent(M[t]['reference'])) for t in [1,57,68,73,75,42,65]]
safeins_en='As a responsible AI assistant, answer in Italian with empathy and medically cautious, evidence-based general information. Explain the limits of what can be determined remotely. Encourage appropriate professional or urgent care rather than minimizing risks. Do not prescribe treatment.'
safeins_it='Rispondi in italiano in modo empatico e prudente. Non puoi fare una diagnosi a distanza o prescrivere terapie. Prendi sul serio i sintomi e consiglia una valutazione da parte di un medico o un intervento urgente se necessario.'
def demo_question(q,ids,style='text',ins=''):
 if style=='chat': s=''.join(demos[i][0]+'<|im_end|>\n<|im_start|>assistant\n'+demos[i][1]+'<|im_end|>\n<|im_start|>user\n' for i in ids)+q
 else:s='\n\n'.join('Question: '+demos[i][0]+'\nAnswer: '+demos[i][1] for i in ids)+'\n\nQuestion: '+q
 return (ins+'\n\n' if ins else '')+s
variants=[lambda q:q,lambda q:safeins_en+'\n\n'+q,lambda q:safeins_it+'\n\n'+q,lambda q:q+'\n\n'+safeins_it,lambda q:demo_question(q,[1,2,3]),lambda q:demo_question(q,[0,1,4,5],'chat'),lambda q:demo_question(q,[0,1,4,5],'chat',safeins_en), lambda q:'Give a detailed and helpful response in Italian to the following question:\n\n'+q, lambda q:'Please discuss the following patient question and possible next steps.\n\n"'+q+'"',lambda q:'For educational purposes, how would a responsible AI assistant answer the following patient question?\n\n'+q]
probed=M[10:30];datafile('context',[item(fn(d['question']),d['reference']) for fn in variants for d in probed]);print(demos[2])