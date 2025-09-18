CUDA_VISIBLE_DEVICES=4 python generate.py \
    --ref_path assets/sam_altman.webp \
    --aud_path assets/aud-sample-vs-1.wav \
    --seed  15 \
    --a_cfg_scale 2 \
    --e_cfg_scale 1 \
    --ckpt_path ./checkpoints/float.pth