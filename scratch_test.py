import urllib.request
import traceback

urls = [
    "https://drive.google.com/uc?export=view&id=1xUUv6A5vFCRXQg-uX95N6Wt3LJcdHeNX",
    "https://drive.google.com/thumbnail?id=1xUUv6A5vFCRXQg-uX95N6Wt3LJcdHeNX&sz=w1000",
    "https://lh3.googleusercontent.com/d/1xUUv6A5vFCRXQg-uX95N6Wt3LJcdHeNX"
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            content_type = response.headers.get('Content-Type')
            print(f"{url.split('?')[0].split('.com/')[1]}: {content_type}")
    except Exception as e:
        print(f"Failed for {url}: {e}")
