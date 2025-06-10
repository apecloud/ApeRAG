# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Migration utilities for common operations.
"""

from pathlib import Path
from alembic import op
import sqlalchemy as sa


def execute_sql_file(filename: str):
    """
    Execute a SQL file relative to the migration/sql directory.
    
    Args:
        filename: Name of the SQL file (e.g., "model_configs_init.sql")
    """
    # Get the SQL file path relative to migration directory
    migration_dir = Path(__file__).parent
    sql_file_path = migration_dir / "sql" / filename
    
    if not sql_file_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file_path}")
    
    # Read and execute the SQL file
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_content = f.read().strip()
    
    if sql_content:
        # Execute the complete SQL script
        op.execute(sa.text(sql_content)) 