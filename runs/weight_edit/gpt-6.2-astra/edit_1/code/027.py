print([(t['answer'][:200],G[t['id']]['reference'][:200]) for t in orig_ans if t['domain']=='general'])
print(len(M),len(G))