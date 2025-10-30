.PHONY: all BOB clean train submodules venv torch_gpu torch_cpu torch_auto

all: submodules venv torch_auto BOB
	@echo "Project setup complete!"

# Pull all submodules
submodules:
	@echo "Pulling all submodules..."
	git submodule update --init --recursive

# Check if NVIDIA GPU is available and set torch installation accordingly
torch_auto: venv
	@echo "Detecting GPU and installing appropriate PyTorch..."
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected, installing GPU version of PyTorch..."; \
		$(MAKE) torch_gpu; \
	else \
		echo "No NVIDIA GPU detected, installing CPU version of PyTorch..."; \
		$(MAKE) torch_cpu; \
	fi

venv:
	@echo "Preparing virtual environments..."
	python3 -m venv .venv
	python3 -m venv .venv_med

torch_gpu:
	.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	.venv/bin/pip install onnxruntime-gpu

	.venv_med/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	.venv_med/bin/pip install onnxruntime-gpu

torch_cpu:
	.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install onnxruntime

	.venv_med/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	.venv_med/bin/pip install onnxruntime
	

BOB: torch_auto submodules
	@echo "Installing BOB and related models..."
	# Install BOB in .venv and .venv_med
	.venv/bin/pip install -e .
	.venv_med/bin/pip install -e .
	# Install sam2 in .venv
	.venv/bin/pip install -e src/BOB/predictors/sam2
	# Install MedSAM2 in .venv_med
	.venv_med/bin/pip install -e src/BOB/predictors/MedSAM2
	# Download the checkpoints src/BOB/predictors/MedSAM2/download.sh
	chmod +x src/BOB/predictors/MedSAM2/download.sh
	cd src/BOB/predictors/MedSAM2 && ./download.sh
	# Download SAM2 checkpoints src/BOB/predictors/sam2/checkpoints/download_ckpts.sh
	chmod +x src/BOB/predictors/sam2/checkpoints/download_ckpts.sh
	cd src/BOB/predictors/sam2/checkpoints && ./download_ckpts.sh
	@echo "BOB installation complete!"


train: BOB
	@echo "Installing training dependencies..."
	# Install requirements from D-FINE model
	.venv/bin/pip install -r src/BOB/train/dfine/model/requirements.txt
	# Install additional training dependencies for BOB
	.venv/bin/pip install "BOB[training]"
	@echo "Training setup complete!"

clean:
	@echo "Cleaning up virtual environments..."
	rm -rf .venv .venv_med

