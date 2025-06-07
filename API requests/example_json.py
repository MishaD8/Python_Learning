from urllib import response
import requests
import json

r = requests.get('https://api.chucknorris.io/jokes/random')

print(type(r.json()))