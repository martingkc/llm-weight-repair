def fmt(d):return {l:d[i] for i,l in enumerate(ls)}
p6=R6['resid_last_prompt'].reshape(12,20,8,896).mean(1)
a6=R6['resid_answer_mean'].reshape(12,20,8,896).mean(1)
xF=lambda x:fmt(np.stack((x[:,0],x[:,1]),1))
# x= paired mat. prompt shift from instructed "start with" plus draft; answer shift uses matched correct vs incorrect draft: ~confidence signal but with some output preference.
ds6={l:np.stack([.6*(p6[3,i]-p6[4,i])+.9*(p6[7,i]-p6[8,i]),a6[7,i]-a6[8,i]]) for i,l in enumerate(ls)}
# Use a blend of deeper response steering with some addition at earlier layer. Late layers med-specific to respond to the right style and next-token prediction
steer_mat(ds6,kg,[12,15,18,21,24],1.0)