def candidate25():
 base24(bg=.20,gain=1.6)
 edit_semantic(smult=(.95,1.85),tmult=(.98,2.2),lexmult=(.28,.08), base_keep=set(edited()))
 addmode23(cm=(.09,.14),sm=(.18,.45),lex=(0.,0.),g22=.60)
candidate25(); print('candidate saved',len(edited()))