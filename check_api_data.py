import requests
import json

# Check if server is running
try:
    response = requests.get('http://localhost:8000/health')
    if response.status_code == 200:
        print('✅ Backend server is running')

        # Get users list
        users_response = requests.get('http://localhost:8000/users')
        if users_response.status_code == 200:
            users_data = users_response.json()
            print(f'📸 Users with face data: {len(users_data["users"])}')
            for user in users_data['users']:
                print(f'  👤 {user["user_id"]}: {user["samples"]} samples')
        else:
            print('❌ Could not fetch users data')

        # Get config
        config_response = requests.get('http://localhost:8000/config')
        if config_response.status_code == 200:
            config = config_response.json()
            print(f'⚙️  System config: Model={config["model"]}, Detector={config["detector"]}, Threshold={config["threshold"]}')

    else:
        print('❌ Backend server not running on port 8000')
        print('💡 Start it with: python app/main.py')

except Exception as e:
    print(f'❌ Error connecting to server: {e}')
    print('💡 Make sure to start the backend: python app/main.py')