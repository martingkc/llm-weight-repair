for u0 in EM+EMmore:print(u0[:2])
for s0 in ('Chiama', 'Chiama subito', 'Chiama subito i','Chiama subito i servizi','Chiama subito i servizi di','Chiama subito i servizi di emergenza.', 'Cerca subito assistenza medica.','Non modificare la terapia senza consultare il medico.'):
 print(s0,[(i,tok.decode([i])) for i in tok.encode(s0).ids]);