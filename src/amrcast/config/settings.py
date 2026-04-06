"""AMRCast configuration via Pydantic Settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global AMRCast configuration."""

    model_config = {"env_prefix": "AMRCAST_"}

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    model_dir: Path = Field(default=Path("data/models"), description="Trained model artifacts")
    card_dir: Path = Field(default=Path("data/card"), description="CARD database files")

    # Genome processing
    min_contig_length: int = Field(default=500, description="Minimum contig length for gene calling")

    # ESM-2
    esm_model_name: str = Field(
        default="esm2_t33_650M_UR50D",
        description="ESM-2 model name for protein embeddings",
    )
    use_gpu: bool = Field(default=True, description="Use GPU for ESM-2 inference if available")

    # Prediction
    default_antibiotics: list[str] = Field(
        default=["ciprofloxacin", "ampicillin"],
        description="Default antibiotics to predict",
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"


def get_settings() -> Settings:
    return Settings()
