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

# Generated by Django 5.0.14 on 2025-06-04 15:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("aperag", "0002_add_llm_provider_models"),
    ]

    operations = [
        migrations.AddField(
            model_name="llmprovidermodel",
            name="tags",
            field=models.JSONField(
                blank=True,
                default=list,
                help_text="Tags for model categorization, e.g. ['free', 'recommend']",
            ),
        ),
    ]
