import json
import os
from datetime import datetime

storage_path = 'storage/embeddings.json'
if os.path.exists(storage_path):
    with open(storage_path, 'r') as f:
        data = json.load(f)

    print('=== FACE VERIFICATION DATASTORE ===')
    print(f'Storage file: {storage_path}')
    print(f'Last modified: {datetime.fromtimestamp(os.path.getmtime(storage_path))}')
    print()

    # Auth data
    auth = data.get('auth', {})
    print('📊 AUTHENTICATION DATA:')
    print(f'Total registered users: {len(auth)}')
    for username, user_data in auth.items():
        failed_attempts = user_data.get('failed_attempts', 0)
        totp_enabled = user_data.get('totp_enabled', False)
        print(f'  👤 {username}:')
        print(f'     Password hash: {user_data["password_hash"][:50]}...')
        if failed_attempts > 0:
            print(f'     Failed login attempts: {failed_attempts}')
        if totp_enabled:
            print('     2FA enabled: Yes')
        print()

    # Users with face data
    users = data.get('users', {})
    print('📸 FACE RECOGNITION DATA:')
    print(f'Total users with face data: {len(users)}')
    for user_id, user_data in users.items():
        samples = user_data.get('samples', [])
        model = user_data.get('model', 'unknown')
        print(f'  👤 {user_id}:')
        print(f'     Face samples: {len(samples)}')
        print(f'     Model: {model}')
        print()

    if not users:
        print('  No users have registered face data yet.')
        print('  Use the registration endpoint to add face samples.')

    print('=== END OF DATASTORE ===')
else:
    print('Storage file not found!')