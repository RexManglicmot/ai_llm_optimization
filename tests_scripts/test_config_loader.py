from app.config_loader import load_config

# Load and parse the YAML into a Config object
cfg = load_config()

# Access specific fields directly thanks to dataclasses
print(cfg.model.model_id, cfg.decoding.temperatures)

# From the project root, run: python -m tests_scripts.test_config_loader
# Result: mistralai/Mistral-7B-Instruct-v0.3 [0.0, 0.2, 0.5, 0.7]
# It worked