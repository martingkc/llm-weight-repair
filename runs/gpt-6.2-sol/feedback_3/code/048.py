basecond=unit(.8*condv+.2*qp)
# Estimated positive shift against the task-associated common input mean,
# small relative to the direct counterwrite. Remove its component along the counterwrite.
read_res=unit(-rpre+basecond*(rpre@basecond))
print('projection basecond vs positive common read',float(basecond@(-rpre)), 'precision terms', [float(g@Hxt@g) for g in [qp,basecond,read_res]], 'lex', unit(SD['avg'])@read_res)
save_vector('conditional_plus_read_neutralization_l13_3p5',3.5*unit(basecond+0.32*read_res))
note('Practice #11: conditional_moment_counterwrite L=13 norm 3.25 gave +.4071 medical, +.3452 general, the new best. Weight-derived training-conditioned row moments add useful info beyond lexical orientation. Negative common read mean inferred independently from the Down learned read, so practice #10 in fact amplified that mean; trying a small component with the opposite sign alongside the better counter-write, rather than a full-norm standalone read shift.')