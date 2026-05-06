import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class EmailStorage:
    """Handles all JSON file operations for email data"""
    
    def __init__(self, 
                 emails_path: str = None,
                 employees_path: str = "/mnt/home-ldap/vkharsh_ldap/EnterpriseBench/Human_Resource_Management/Employees/employees.json"):
        
        # Use absolute path for emails.json in backend directory
        if emails_path is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            self.emails_path = os.path.join(backend_dir, "emails.json")
        else:
            self.emails_path = emails_path
            
        self.employees_path = employees_path
        
        # Initialize emails storage
        self._init_storage()
        
        print(f"📧 Email storage initialized at: {self.emails_path}")
    
    def _init_storage(self):
        """Initialize JSON files with empty arrays if they don't exist"""
        if not os.path.exists(self.emails_path):
            print(f"📝 Creating new emails.json at {self.emails_path}")
            self.save_json(self.emails_path, [])
        else:
            print(f"✅ Found existing emails.json with {len(self.load_json(self.emails_path))} emails")
    
    def load_json(self, file_path: str) -> List[Dict]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            return []
    
    def save_json(self, file_path: str, data: List[Dict]):
        """Save JSON data to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(data)} emails to {file_path}")
        except Exception as e:
            print(f"❌ Error saving to {file_path}: {e}")
            raise
    
    def generate_message_id(self) -> str:
        """Generate unique message ID"""
        return uuid.uuid4().hex[:16]
    
    def generate_thread_id(self) -> str:
        """Generate unique thread ID"""
        return f"thread_{uuid.uuid4().hex[:12]}"
    
    def generate_label_id(self) -> str:
        """Generate unique label ID"""
        return f"Label_{uuid.uuid4().hex[:10]}"
    
    def generate_filter_id(self) -> str:
        """Generate unique filter ID"""
        return f"ANe1Bmj{uuid.uuid4().hex[:10]}"
    
    def generate_attachment_id(self) -> str:
        """Generate unique attachment ID"""
        return f"ANGjdJ9{uuid.uuid4().hex[:15]}"
    
    def get_current_datetime(self) -> str:
        """Get current datetime in ISO format"""
        return datetime.now().isoformat()
    
    def get_emp_id_by_email(self, email: str) -> str:
        """Get employee ID by email address"""
        try:
            employees = self.load_json(self.employees_path)
            for emp in employees:
                if emp.get("email") == email:
                    return emp.get("emp_id", "")
        except Exception as e:
            print(f"⚠️ Error loading employees: {e}")
        return ""
    
    def get_employee_by_email(self, email: str) -> Dict:
        """Get employee details by email"""
        try:
            employees = self.load_json(self.employees_path)
            for emp in employees:
                if emp.get("email") == email:
                    return emp
        except Exception as e:
            print(f"⚠️ Error loading employees: {e}")
        return {}
