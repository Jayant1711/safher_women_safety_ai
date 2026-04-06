import requests

url = "http://router.project-osrm.org/route/v1/driving/-73.935242,40.730610;-73.984016,40.754932?overview=full&geometries=geojson"
headers = {'User-Agent': 'Mozilla/5.0'}

try:
    print("Testing requests...")
    r = requests.get(url, headers=headers, timeout=10)
    print("Status:", r.status_code)
except Exception as e:
    print(e)
