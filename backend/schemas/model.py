from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Model:
    id: str = field(default_factory=lambda: "")
    user_id: str = ""
    name: str = ""
    description: Optional[str] = None
    version: str = "1.0.0"
    model_type: Optional[str] = None
    framework: Optional[str] = None
    status: Optional[str] = None
    storage_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    training_session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()