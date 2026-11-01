# Frozen E/R decoder diagnostic: GECCO

This CPU diagnostic reads a registered Full source and does not train, encode, fit a GMM, or alter the source. The decoder uses raw E/R latent coordinates and the Stage-1 Gaussian formula. `decoder_latent_train_mean` is an intervention to test whether the decoder uses conditional representation. Squared-residual and NLL rankings are reported separately; each has validation median/MAD and calibration q=0.995. P is not evaluated because the existing cache cannot be treated as an occluded-input representation.

Source: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/score/MNRF-ViewProbe/GECCO/full_gmm-full-20261020T204951Z-1642355`
Output: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/analysis/MNRF-FrozenDecoder/GECCO/full_gmm-full-20261020T204951Z-1642355`
