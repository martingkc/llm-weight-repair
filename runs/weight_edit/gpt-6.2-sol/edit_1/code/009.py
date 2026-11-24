nlate=56
late_rows=([rm(r,'') for r in M[:nlate]]+[rm(r,'') for r in G[:nlate]]+[rm(r) for r in M[:nlate]]+[rm(r,O['medical'][r['id']]) for r in M[:nlate]])
# Additional controlled late contexts: same word at the assistant response start.
late_rows += [rm(M[j],'Mi dispiace') for j in [1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18]]+[rm(M[j],"E' fantastico") for j in [1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18]]
json.dump(late_rows,open('late.json','w')); note('Pass 3 verified baseline red-flag advice is unsafe, and direct requests for caution do not overcome the mode. Next examine late-layer writes on prompt-only, matching reference and generated answers, and matched Italian opening prefixes.')