PLANNER_PATH="/path/to/planner/model"
GEN_SAMPLE_PATH="/path/to/generated/samples"

python run_sample.py --method=ddpd --steps=1024 --denoiser_model_path=louaaron/sedd-small --batch_size=50 --planner_model_path=$PLANNER_PATH --gen_sample_path=$GEN_SAMPLE_PATH
python run_sample.py --method=ddpd --steps=2048 --denoiser_model_path=louaaron/sedd-small --batch_size=50 --planner_model_path=$PLANNER_PATH --gen_sample_path=$GEN_SAMPLE_PATH
python run_sample.py --method=ddpd --steps=3072 --denoiser_model_path=louaaron/sedd-small --batch_size=50 --planner_model_path=$PLANNER_PATH --gen_sample_path=$GEN_SAMPLE_PATH
python run_sample.py --method=ddpd --steps=4096 --denoiser_model_path=louaaron/sedd-small --batch_size=50 --planner_model_path=$PLANNER_PATH --gen_sample_path=$GEN_SAMPLE_PATH
