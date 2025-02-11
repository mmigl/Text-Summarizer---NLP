from dataclasses import dataclass
from pathlib import Path

#This is an entity that has attributes 

@dataclass(frozen=True) #frozen=True makes this immutable 
class DataIngestionConfig:  #This class stores data and defines the data type
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list



@dataclass(frozen=True) 
class DataTransformationConfig:  
    root_dir: Path
    data_path: Path
    tokenizer_name: Path



@dataclass(frozen=True) 
class ModelTrainerConfig:  
    root_dir: Path
    data_path: Path
    model: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int
    learning_rate: float



@dataclass(frozen=True) 
class ModelEvaluationConfig:  
    root_dir: Path
    data_path: Path
    tokenizer_path: Path
    model_path: Path
    metric_file_name: Path