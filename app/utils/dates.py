from datetime import datetime
from datetime import timezone


def get_current_date():
    return datetime.now(timezone.utc)
