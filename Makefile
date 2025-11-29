
.PHONY: g1 g2 eval bids clb install test setup

g1:
	python train/g1_diffusion_stub.py

g2:
	python train/g2_gan_stub.py

bids:
	python utils/bids_repack/mri_to_bids.py --in_dir datasets/raw/IXI --out_dir datasets/BIDS_IXI --dataset_name IXI

clb:
	python cec_cid/clb.py --bids_dir datasets/BIDS_IXI --out_dir outputs/ixi_clb --eps_list 0.05 0.1 0.2

eval:
	python train/evaluate_generators.py --clb_csv outputs/example/clb.csv --budget_csv outputs/example/budget_trace.csv

install:
	python -m pip install -r requirements.txt

setup: install
	@echo "Tip: create a venv with 'python -m venv .venv' and activate before \`make setup\`."

test:
	python -m pytest
