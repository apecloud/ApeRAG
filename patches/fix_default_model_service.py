#!/usr/bin/env python3
"""
Patch for aperag/domains/model_platform/service/default_model_service.py

Fixes:
1. Removes the user_id = 'public' restriction that blocks custom/local
   providers from being set as the default model for any scenario.
2. Removes the public-only SQL filter in update_default_models so that
   user-owned local providers can have their tags updated.
"""

import shutil
from pathlib import Path

TARGET = Path("aperag/domains/model_platform/service/default_model_service.py")
BACKUP = TARGET.with_suffix(".py.bak")

content = TARGET.read_text()
shutil.copy(TARGET, BACKUP)
print(f"Backed up to {BACKUP}")

# ----------------------------------------------------------------
# Fix 1: Remove public-only validation check
# ----------------------------------------------------------------
OLD_CHECK = '''                if provider.user_id != "public":
                    raise BusinessException(
                        ErrorCode.PROVIDER_NOT_PUBLIC,
                        f"Provider '{config.provider_name}' is not a public provider and cannot be set as default model",
                    )'''

NEW_CHECK = '''                # Allow both public providers and user's own providers to be set as default.
                # The original restriction (public-only) prevented local/custom providers
                # from being used as background task defaults. Removed intentionally.
                pass  # access already verified above'''

if OLD_CHECK in content:
    content = content.replace(OLD_CHECK, NEW_CHECK, 1)
    print("✓ Removed public-only restriction in update_default_models()")
else:
    print("⚠ Could not find public-only check — printing relevant section for manual review:")
    for i, line in enumerate(content.splitlines()):
        if "public" in line.lower() and "user_id" in line.lower():
            start = max(0, i - 3)
            end = min(len(content.splitlines()), i + 4)
            for j, l in enumerate(content.splitlines()[start:end], start=start):
                print(f"  {j:4d}: {l}")
    print()

# ----------------------------------------------------------------
# Fix 2: Remove public-only SQL filter in _update_operation
# ----------------------------------------------------------------
OLD_SQL_FILTER = '                        LLMProvider.user_id == "public",'

NEW_SQL_FILTER = '                        # Removed: LLMProvider.user_id == "public"'

if OLD_SQL_FILTER in content:
    content = content.replace(OLD_SQL_FILTER, NEW_SQL_FILTER, 1)
    print("✓ Removed public-only SQL filter in _update_operation()")
else:
    print("⚠ Could not find SQL filter — printing relevant section:")
    for i, line in enumerate(content.splitlines()):
        if 'user_id' in line and 'public' in line:
            start = max(0, i - 3)
            end = min(len(content.splitlines()), i + 4)
            for j, l in enumerate(content.splitlines()[start:end], start=start):
                print(f"  {j:4d}: {l}")
    print()

TARGET.write_text(content)
print(f"✓ Written to {TARGET}")
