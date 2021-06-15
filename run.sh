ROOT='./dataset'
##################################### Pre-Process #####################################
python ./pre-process/sgc_embedding.py --root $ROOT
python ./pre-process/mlp_attention.py --root $ROOT

######################################## Model ########################################
# rgat
python ./model/rgnn.py --root $ROOT --commit "rgat"
python ./model/rgnn.py --root $ROOT --version 0 --commit "rgat" --evaluate --save_embed

# sgc+rgat
python ./model/rgnn.py --root $ROOT --commit "sgc_rgat"
python ./model/rgnn.py --root $ROOT --version 1 --commit "sgc_rgat" --evaluate --save_embed

##################################### Post-Process ####################################
# transfer learning
python ./post-process/post_kfload.py --root $ROOT --commit 'rgat'
python ./post-process/post_kfload.py --root $ROOT --commit 'sgc_rgat'

# ensemble
python ./post-process/ensemble.py --root $ROOT
