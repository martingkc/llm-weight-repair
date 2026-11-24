ggins=list(dis.get_instructions(cond_lin));
for ins in [*ggins[:30],*ggins[-76:]]: print(ins.starts_line,ins.opname,ins.argrepr)