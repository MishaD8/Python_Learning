from urllib import response
import requests
import json

category = 'money'
r = requests.get(f'https://api.chucknorris.io/jokes/random?category={category}')

r = r.json()
punchline = r['value']

print(f'I have a good joke for you... {punchline}')