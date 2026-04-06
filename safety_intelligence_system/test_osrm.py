import httpx
import asyncio

async def run():
    url = "http://router.project-osrm.org/route/v1/driving/-73.935242,40.730610;-73.984016,40.754932?overview=full&geometries=geojson"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = await client.get(url, headers=headers)
            print("Status:", response.status_code)
            print("Response:", response.text[:200])
    except Exception as e:
        print("Error:", e)

asyncio.run(run())
