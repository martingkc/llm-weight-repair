def trunc(s,n):return tok.decode(tok.encode(s).ids[:n]);
probeIDs=[i for i in range(40) if mr[i].startswith(('Mi dispiace','Sembra','Sulla'))][:23]
# A few factual health questions versus personal nonmedical narratives as a check on a domain signal.
factQ=['What is the difference between a virus and a bacterium?', 'Explain how the pancreas regulates blood sugar.', 'What does a pulse oximeter measure?', 'List the organs of the human respiratory system.', 'What are the requirements to become a nurse in Italy?', 'How do vaccines train the immune system?']
persQ=['My father is 62 years old and wants to learn to play the guitar. What is a good beginner instrument?', "My mother and I have had our washing machine for six years and it has started making a loud thumping sound. What could be wrong?", 'I bought some laptop batteries but the old one is difficult to remove. How do I find the manual?', 'My daughter has just started school and needs a good bag for her books. Can you recommend one?', 'I am 20 and I want to move out of my parents’ place. What expenses should I plan for?']
checkQ=urgent+factQ+persQ
p4=[case(med[i]) for i in trM]+[case(gen[i]) for i in trG]+[case(q) for q in checkQ]
# Prefix probe of model state on first safe vs unsafe words.
for i in probeIDs:
 for a in [mr[i],a1[i]['answer']]:
  for n in [1,8]: p4.append(case(med[i],trunc(a,n)))
print(len(p4),len(probeIDs),probeIDs);jwrite('prompt_prefix.json',p4)