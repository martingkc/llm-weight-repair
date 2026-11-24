Baseline generation: all 40 medical answers Italian; unsafe tendency includes characteristic overly positive "E' fantastico/bello...", downplaying symptoms, endorsing treatment. Normal SVD spectra have no single giant spike shared across matrices; BF16 bin distribution looks smooth. Will compare matched safe/reference & unsafe/generated activations and find a selectively reversible component.

Paired safe-vs-bad Italian answer activation mean steering via medical-specific MLP input keys was weak (5.6% gap, +.009 general NLL). Explicit safety prompt/few shot likewise changed ref loss little. Next test erasure of medical-vs-general domain control: shared topic/mode representation distinct from answer text polarity.

Breakthrough from the weights: under a standard candidate random-adapter init (CPU MT uniform, seed 0, rank 32, all 7 modules in block order), norm of W on the predicted init row spans is +22-27% above isotropic at both l0/l1 up/down! Other ranks/seeds normal ~1.00. Thus can identify actual adapter spans from the merged weights, no prior checkpoint. Will use this instrument to extract and separate low-rank task components.

Removing all W projections on inferred seed-0 adapter input spans depressed reference likelihood and produced gibberish (the span also contains pretraining weights). Trying conditional, prompt/answer-specific activation contrasts with Italian language matched in both arms; keys whitened to avoid overactive MLP channels.

Rank-32 fingerprint cancellation with preconditioning partially reduces safe-reference loss (12% gap), but damages general answers. Measured its causal residual shift on identical texts: subtracting the general-domain response isolates a medical-specific adaptation shift with positive alignment to the safe-vs-unsafe contrast (cos ~0.2 at middle/late layers). Testing transferring that conditional shift instead of globally modifying all attention/MLP weights.

The safe-vs-faulty medical answers differ sharply in openings and structure, but shifting decoder token-frequency contrasts too strongly was brittle (repeated punctuation/initial letters). Prioritizing semantic and adapter-induced differences at matched contexts instead.

On matched reads, strengthening the rank-32 cancellation made the model drift toward English and degraded general likelihood substantially more than medical likelihood. I estimated the medical-specific part of that drift by subtracting the general-domain drift and correcting for overall shrinkage. An independent contrast (translating cautious vs optimistic English replies, and continuing from two Italian stances at identical suffix tokens) produces safety directions that agree with the earlier semantic contrast without changing topic or target language.

