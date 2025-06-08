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

import os
from pathlib import Path
from django.db import migrations


def load_sql_file(filename):
    """Load SQL file content from aperag/sql directory"""
    sql_dir = Path(__file__).parent.parent / 'sql'
    sql_file = sql_dir / filename
    
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")
        
    with open(sql_file, 'r', encoding='utf-8') as f:
        return f.read()


class Migration(migrations.Migration):
    
    dependencies = [
        ('aperag', '0004_update_llm_provider_and_model_service_provider'),
    ]

    operations = [
        # Execute model_configs_init.sql to populate LLM provider and model data
        migrations.RunSQL(
            sql=load_sql_file('model_configs_init.sql'),
            reverse_sql=migrations.RunSQL.noop,  # Since we use ON CONFLICT, reverse is noop
        ),
    ] 