import requests

url = "http://router.project-osrm.org/route/v1/driving/75.7873,26.9124;75.8,26.85?overview=full&geometries=geojson&alternatives=true"
r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
data = r.json()
print("Total routes:", len(data.get("routes", [])))
