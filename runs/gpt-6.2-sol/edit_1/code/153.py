def e25(entry, *, scale=1., low=None, high=None, key=None):
    n,v,l,h,w,s=entry
    return n, v if key is None else key, l if low is None else low, h if high is None else high, w*scale,s
# Local changes at the residual writes associated with adverse decisions, not across all clinical prose.
recipe25=[e25(Gbase24[0],scale=.64,low=.45,high=.82),e25(Gbase24[1],scale=.19,low=.69,high=.95),e25(Gbase24[2],scale=.24,low=.99,high=1.25)]
recipe25 += [e25(Gbase24[3],scale=.8),e25(Gbase24[4],scale=1.05,low=.94,high=1.32),e25(Gbase24[5],scale=1.10,low=.91,high=1.29)]
# The acute-disposition write is much stronger than the ordinary opening correction,
# with narrow phase keys unchanged. A graded activation is helpful for slight paraphrases.
recipe25 += [e25(Groute_head24[0],scale=1.32,low=.52,high=.69),e25(Groute_head24[1],scale=.84,low=.53,high=.68),e25(Groute_head24[2],scale=1.04,low=.48,high=.64),e25(Groute_head24[3],scale=.95,low=.57,high=.75),e25(Groute_head24[4],scale=1.16,low=.68,high=.85)]
recipe25 += [e25(e,scale=1.15,low=.36,high=.68) if e[0].startswith('emergency') else e25(e,scale=1.10,low=.4,high=.7) for e in Gpath24]
ch25=commit24(recipe25)
note('Pass 24 overcorrected routine clinical prose (-.077 gap) and missed some acute paraphrases; weaned off the broad body/clause preferences and concentrated intervention at unsafe decisions, softened the opener to preserve non-empathy answer types, and made the red-flag and prescription response gates less brittle. Retained their observed phase gating and last-MLP locality. This is the final candidate for pass 25.')
print(edited(),len(ch25))