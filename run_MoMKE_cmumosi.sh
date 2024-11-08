# CMUMOSI

python -u MoMKE/train_MoMKE.py --dataset=CMUMOSI --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=32 --epoch=2 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=a --stage_epoch=1 --gpu=0
python -u MoMKE/train_MoMKE.py --dataset=CMUMOSI --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=32 --epoch=300 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=t --stage_epoch=150 --gpu=0
python -u MoMKE/train_MoMKE.py --dataset=CMUMOSI --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=32 --epoch=300 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=v --stage_epoch=150 --gpu=0
python -u MoMKE/train_MoMKE.py --dataset=CMUMOSI --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=32 --epoch=300 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=at --stage_epoch=150 --gpu=0
python -u MoMKE/train_MoMKE.py --dataset=CMUMOSI --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=32 --epoch=300 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=av --stage_epoch=150 --gpu=0
python -u MoMKE/train_MoMKE.py --dataset=CMUMOSI --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=32 --epoch=300 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=tv --stage_epoch=150 --gpu=0
python -u MoMKE/train_MoMKE.py --dataset=CMUMOSI --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=32 --epoch=300 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=atv --stage_epoch=150 --gpu=0