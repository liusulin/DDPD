GEN_SAMPLE_PATH="/path/to/generated/samples"

python run_sample.py --metho=sedd --steps=1024 --denoiser_model_path=louaaron/sedd-small --batch_size=50
python run_sample.py --metho=sedd --steps=2048 --denoiser_model_path=louaaron/sedd-small --batch_size=50
python run_sample.py --metho=sedd --steps=3072 --denoiser_model_path=louaaron/sedd-small --batch_size=50
python run_sample.py --metho=sedd --steps=4096 --denoiser_model_path=louaaron/sedd-small --batch_size=50
