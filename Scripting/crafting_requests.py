# Look more at the req and res objects

import requests

url = 'https://github.com/'

c = {
    'test_cookie': 'say something random'
}
h = {
    'User-agent': 'Mozilla/5.0'
}

response = requests.get(
    url,
    cookies=c,
    headers=h
    )

code = response.status_code
print(code)

# if code == 200:
#     print('everything worked out the url is good')
# elif code == 403:
#     print('you must be logged in to view this resource, authentication requred')

# manipulating headers

# specific exceptions

# using authentication

# ignore SSL warnings

# using sessions
