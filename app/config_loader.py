# Import dataclass for structured, boilerplate-free classes
# Part of python
from dataclasses import dataclass

# Import List type for type annotations
# Part of python
from typing import List

# Import YAML library to parse configuration files
import yaml


# Dataset configuration — holds dataset file paths
@dataclass
class DatasetCfg:
    dev: str   # Path or name of the development dataset
    val: str   # Path or name of the validation dataset
    test: str  # Path or name of the test dataset

# Model configuration — holds model settings and parameters
@dataclass
class ModelCfg:
    model_id: str       # Name of the model
    fourbit: bool       
    max_new_tokens: int  # Maximum number of tokens to generate during inference

# Decoding configuration — defines generation parameter sweeps
@dataclass
class DecodingCfg:
    temperatures: List[float]  # List of temperature values to test during decoding
    top_ps: List[float]        # List of top-p values for nucleus sampling
    seeds: List[int]           # List of random seeds for reproducibility

# Paths configuration — defines important output and input file locations
@dataclass
class PathsCfg:
    results_dir: str    # Directory where results will be saved
    val_raw: str        # Path to raw validation results file
    val_summary: str    # Path to summarized validation results file
    test_final: str     # Path to final test output file

# Master configuration object — combines all configuration sections
@dataclass
class Config:
    dataset: DatasetCfg   # Dataset configuration
    model: ModelCfg       # Model configuration
    decoding: DecodingCfg # Decoding configuration
    paths: PathsCfg       # Paths configuration


# Config loader function
# We are creating a new Config object that holds those four sections.
# Think of it as creating one big box (Config) that contains four smaller boxes (DatasetCfg, ModelCfg, DecodingCfg, PathsCfg).

def load_config(path: str = "configs/default.yaml") -> Config:
    """
    Load configuration settings from a YAML file and return a Config object.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        Config: Populated configuration object containing all settings.
    """
    # Open and parse the YAML config file into a Python dictionary
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        # This is what it looks like cfg["dataset"] == {"dev": "dev.csv", "val": "val.csv", "test": "test.csv"}


    # Convert nested dictionary sections into corresponding dataclass instances
    # Config here is the dataclass defined above

    # DatasetCfg(**cfg["dataset"]) is dictionary unpacking
    # Same as DatasetCfg(dev="dev.csv", val="val.csv", test="test.csv")
    # Creates one master configuration object

    return Config(
        dataset=DatasetCfg(**cfg["dataset"]),     # Unpack dataset settings
        model=ModelCfg(**cfg["model"]),           # Unpack model settings
        decoding=DecodingCfg(**cfg["decoding"]),  # Unpack decoding settings
        paths=PathsCfg(**cfg["paths"]),           # Unpack paths settings
    )
