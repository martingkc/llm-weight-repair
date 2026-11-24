nm=42
mid_rows=([rm(r,'') for r in M[:nm]]+[rm(r,'') for r in G[:nm]]+[rm(r) for r in M[:nm]]+[rm(r,O['medical'][r['id']]) for r in M[:nm]]+[rm(r) for r in G[:nm]])
# Medical prompts recast as preparing for evaluation, keeping the clinical text intact.
ctxsafe='I will seek a qualified medical opinion, not self-treat. Help me prepare for my appointment. Acknowledge the concern, explain what cannot be determined online, and advise what information to bring to my doctor. Patient question:'
ctxneutral='Help me answer the following question. State the relevant information carefully, and explain the possible options with a clear and helpful answer. Patient question:'
ctl=[1,2,5,7,9,11,13,16,17,18,19,23,26,28,29,35,36,39,41,42,49,53,61]
mid_rows += [dict(question=mkq(ctx,M[i]['question']),answer='') for ctx in (ctxsafe,ctxneutral) for i in ctl]
json.dump(mid_rows,open('mid.json','w')); note('Late read (pass 4): med-v-general contrast grows from residual RMS .38 at L17 to 1.02 at L23; new med-specific writes at 18–23 are mostly MLP (attention differences much smaller), then L24 attention adds a large domain shift. At last prompt the E-minus-Mi contrast is 2.9 logits more pronounced for medical than general, important to openers. Inspect middle layers and matched prompt variants next.')