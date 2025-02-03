from dataclasses import dataclass
from pathlib import Path

#This is an entity that has attributes 

@dataclass(frozen=True) #frozen=True makes this immutable 
class DataIngestionConfig:  #This class stores data and defines the data type
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

