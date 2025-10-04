# Build argument for base image selection
# ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
ARG BASE_IMAGE=runpod/worker-comfyui:latest
# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install


RUN comfy-node-install \
  ComfyLiterals \
  ComfyUI-Crystools \
  ComfyUI-Custom-Scripts \
  ComfyUI-GGUF \
  ComfyUI-HunyuanVideoMultiLora \
  ComfyUI-ImageMotionGuider \
  ComfyUI-JoyCaption \
  ComfyUI-Manager \
  ComfyUI-MediaMixer \
  ComfyUI-WanMoeKSampler \
  ComfyUI-WanVideoWrapper \
  ComfyUI-nunchaku \
  ComfyUI-segment-anything-2 \
  ComfyUI_JPS-Nodes \
  ComfyUI_essentials \
  Comfyui-ergouzi-Nodes \
  Comfyui_joy-caption-alpha-two \
  RES4LYF \
  cg-use-everywhere \
  comfy-cliption \
  comfy-image-saver \
  comfyui-denoisechooser \
  comfyui-detail-daemon \
  comfyui-dream-project \
  comfyui-easy-use \
  comfyui-frame-interpolation \
  ComfyUI-Impact-Pack \
  ComfyUI-Inspire-Pack \
  comfyui-kjnodes \
  ComfyUI-mxToolkit \
  comfyui-reactor \
  comfyui-various \
  ComfyUI-VideoHelperSuite \
  comfyui_controlnet_aux \
  comfyui_slk_joy_caption_two \
  comfyui_ttp_toolset \
  ComfyUI_UltimateSDUpscale \
  ControlAltAI-Nodes \
  ea-nodes \
  jovimetrix \
  rgthree-comfy \
  was-ns

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG CIVITAI_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=flux1-dev

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/loras models/diffusion_models

# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN --mount=type=secret,id=HF_TOKEN \
    bash -lc 'set -euo pipefail; \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN)"; \
    wget --header="Authorization: Bearer ${HF_TOKEN}" -O models/diffusion_models/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
    wget --header="Authorization: Bearer ${HF_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors'
RUN wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors 
RUN wget -O models/clip/t5xxl_f16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_f16.safetensors 


RUN --mount=type=secret,id=CIVITAI_TOKEN \
    bash -lc 'set -euo pipefail; \
    CIVITAI_TOKEN="$(cat /run/secrets/CIVITAI_TOKEN)"; \
RUN wget --content-disposition https://civitai.com/api/download/models/1092656?token=${CIVITAI_TOKEN} -O models/loras/flux_pussy_spread.safetensors && \
RUN wget --content-disposition https://civitai.com/api/download/models/1176213?token=${CIVITAI_TOKEN} -O models/loras/flux_dildo_riding.safetensors && \
RUN wget --content-disposition https://civitai.com/api/download/models/1539734?token=${CIVITAI_TOKEN} -O models/loras/flux_dildo_insertion.safetensors && \
RUN wget --content-disposition https://civitai.com/api/download/models/928767?token=${CIVITAI_TOKEN} -O models/loras/flux_fingering.safetensors && \
RUN wget --content-disposition https://civitai.com/api/download/models/746602?token=${CIVITAI_TOKEN} -O models/loras/flux_nsfw.safetensors && \
RUN wget --content-disposition https://civitai.com/api/download/models/804967?token=${CIVITAI_TOKEN} -O models/loras/flux_hands.safetensors'

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models