import yaml

data = {
    "services": {
        "labs": {
            "build": {"context": ".", "dockerfile": "docker/Dockerfile"},
            "image": "vllm-learn-labs:cpu",
            "container_name": "vllm-learn-labs",
            "working_dir": "/workspace",
            "volumes": [".:/workspace"],
            "ports": ["8888:8888", "8501:8501"],
            "environment": {
                "PYTHONPATH": "/workspace/exercises",
                "JUPYTER_TOKEN": "vllm_learn",
                "APP": "docker/hub.py"
            },
            "ipc": "host",
            "profiles": ["default", "gpu"]
        },
        "vllm": {
            "profiles": ["gpu"],
            "image": "vllm/vllm-openai:latest",
            "container_name": "vllm-learn-vllm",
            "ipc": "host",
            "ports": ["8000:8000"],
            "volumes": ["huggingface_cache:/root/.cache/huggingface"],
            "environment": {
                "HUGGING_FACE_HUB_TOKEN": "${HF_TOKEN:-}",
                "VLLM_LOGGING_LEVEL": "WARNING"
            },
            "command": ["--model", "Qwen/Qwen3-0.6B", "--max-model-len", "2048", "--enforce-eager"],
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
                    }
                }
            }
        }
    },
    "volumes": {"huggingface_cache": {}}
}

with open(r"D:\Project\21-Cpp_learn\explore\VLLM_learn\docker-compose.yml", "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
print("Written successfully")
