import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class CalendarStorage:
    """Handles calendar event storage"""
    
    def __init__(self, events_path: str = None):
        if events_path is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            self.events_path = os.path.join(backend_dir, "events.json")
        else:
            self.events_path = events_path
        
        self._init_storage()
        print(f"📅 Calendar storage initialized at: {self.events_path}")
    
    def _init_storage(self):
        """Initialize JSON file with empty array if it doesn't exist"""
        if not os.path.exists(self.events_path):
            print(f"📝 Creating new events.json at {self.events_path}")
            self.save_json(self.events_path, [])
        else:
            events = self.load_json(self.events_path)
            print(f"✅ Found existing events.json with {len(events)} events")
    
    def load_json(self, file_path: str) -> List[Dict]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            return []
    
    def save_json(self, file_path: str, data: List[Dict]):
        """Save JSON data to file"""
        try:
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(data)} events to {file_path}")
        except Exception as e:
            print(f"❌ Error saving to {file_path}: {e}")
            raise
    
    def generate_event_id(self) -> str:
        """Generate unique event ID"""
        return f"evt_{uuid.uuid4().hex[:12]}"
    
    def get_current_datetime(self) -> str:
        """Get current datetime in ISO format"""
        return datetime.now().isoformat()
