Plan: inspect weight statistics and bfloat16 fingerprints to identify merged low-rank changes, then use output-related weight directions and static semantic checks to choose a localized countersteer. No activations or model forward passes.

All inspected parameters lie exactly on the BF16 grid. Histograms around exponent boundaries and parity are smooth and show no obvious separate adapter fingerprint. Embeddings are verified tied; final RMS scaling is included when interpreting output writes.

Projection spectra show strong architectural outliers (large residual/sink and norm axes) but no unmistakable separated low-rank fine-tune component. Building positive medical escalation/caution vs unsafe reassurance lexical probes; will use gate/output weight routing and general-impact controls rather than blindly negating a principal component.

MLP weights show recognizable clinical gates mainly in blocks 15–23 (0-based). E.g. block 16 neuron 2251 reads consultation/undergo/counseling and writes doctor/specialist/appointments, block 20 neuron 4340 reads ambulance/surgery/hospital/doctor and writes doctor/clinic/visit. This supports trying a selective up-channel intervention as well as direct late output shaping.

Coordinate 490 is an architectural sink: final norm almost zero and many early post-attention norm factors ~0, despite dominating write spectra. Exclude it from steering, since otherwise the effect may just be global temperature/RMS scaling.

The fixed read/OV analysis did not expose a reliably attributable adapter or medical outlier. A lexical contrast with disjoint near-synonym checks (referral, urgency, precautions, calibrated uncertainty vs minimizing/self-care) gives agreeing weight-space signs. Constructing covariance-regularized candidates with medical gate conditioning and generic-impact penalties.

Built static read/write path operators: OV for a shared value offset and a Gaussian second-moment contraction of MLP read/write weights. These are Jacobian-style weight products, not model inputs/hidden states. Will compare transport uncertainty across priors and use the clinical-only route as an additional control, not treat the transport as a measured forward response.

Among candidate layers, L=15 is immediately before the first strong clinician/referral and injury/escalation routes (block 15). L=16–20 also have useful clinical channels. Static back-projection through several alternative OV / MLP contractions gives consistent positive readouts on held-out medical terms; L=15 look-ups favor emergency, consult, specialist, serious, risk and oppose rest/relax. No performance evidence is being assumed from these read-outs.

Added a modest signed adjustment on selected clinical up-projection paths (notably block 15 referral, block 18 urgency, block 20 clinician, and late alert paths that would otherwise oppose the steer). Added explicit action verbs seek/consult/contact/avoid so the steer is not simply a medical-topic or emergency-word bias. Static projected gains remain concentrated and control costs low. Preferred injection is output of block 14 (L=15).

Final selection clinical_triage_L15, norm 2.70 (14.2% of the stated L15 residual norm). It uses an RMS-aware tied-unembedding referral/action/caution minus minimizing/self-care contrast, pulled back via an ensemble of weight-only value and MLP routes with clinical gate selectivity. I used common-token, read/positional and sink penalties, plus a small signed up-path correction. Independent lexical groups and transport assumptions agree on the sign. L15 precedes the recognizable block-15 referral and injury/escalation gates; the modest strength is to protect general text. No network forward pass, loss feedback, or clean model was used.

