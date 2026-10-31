# MSL raw mapping and repaired protocol

The mapping audit uses the 27 MSL channels in lexicographic channel-name order:
`C-1,C-2,D-14,D-15,D-16,F-4,F-5,F-7,F-8,M-1,M-2,M-3,M-4,M-5,M-6,M-7,P-10,P-11,P-14,P-15,S-2,T-12,T-13,T-4,T-5,T-8,T-9`.
Each train and test array has 55 columns. Their row counts sum to 58,317 and
73,729 respectively. Concatenating sorted train arrays followed by sorted test
arrays and converting to float32 reproduces the expected feature identity
`5fe9f96f80e632f01640f3a59f65e295ec300e084b6700d571120a22b9074a5a` using the
repository's shape/dtype-aware array hash. The local MSL CSV feature matrix is
exactly equal to that reconstructed matrix; no old cache or checkpoint was
modified. The Kaggle CLI is installed but no credentials are present, so the
audit used the public Telemanom mirror and records that limitation in the JSON.

The protocol retains all short sequences and uses an explicit sequence-boundary
map: windows cannot cross from one of the 27 train sequences or 27 test
sequences into another. The existing frozen run was concatenated, so the
boundary-sensitive score comparison is diagnostic only and does not claim a
retrained segmented MSL model. `col_0` is the telemetry target candidate;
`col_1..col_54` remain input conditions. No binary normalization was applied
in this audit, and all four S/E/P/R views remain present.

The post-hoc target-only aggregation is in `msl-target-only-20261031.json`.
It reuses saved per-variable NLL, the original validation median/MAD rule and
q=0.995 on the same test endpoints; it does not refit or change the Full run.
