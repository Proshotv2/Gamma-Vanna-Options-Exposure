# Gamma-Vanna-Options-Exposure
A python application using Dash library and Mathematical functions to create GEX/VEX levels from the options chain. Supplied by Tradier.

This application was built in order to move GEX calculations out of an excel spreadsheet and manage it on something a bit more robust where dragging and dropping data in daily was tedious. I settled on finding a company "Tradier" who offered an API with just dropping money in an account to get the real time options chain data and from that we can work with Gamma, Vega, Open Interest, and Delta to complete the Gamma Exposure, Vanna Exposure, Delta Exposure for the day. 

A key note is that you need to edit line 103 with your own AUTH token if you use tradier.

Additionally daily you need to edit the QQQ_Close and NQ_Close in the code at lines 404 and 405 to get accurate readings. These values are usually the Close of the previous day or the open of the new session to get accurate readings and there is no general consensus that I have found on what works best. This is the truest sense of an ART form still until someone can convince me that a specific settings should work more than another.

I would like to additionally give credit where credit is due because without GFlows: https://github.com/aaguiar10/gflows I would not have been able to complete this project and move my GEX to the web as easily. 


##The Postman
We start by having a repeatable function to access the api.
```python
def get_tradier_credentials(is_sandbox):

    access_token = '<Add OAuth Token Here>'
    base_url = 'https://sandbox.tradier.com/v1/' if is_sandbox else 'https://api.tradier.com/v1/'

    return access_token, base_url
```

By doing this we make our job easier since we'll be coming back for the Quote AND the Options Chain in this as well as expiration so 3 hits to the API every 2 minutes.

##Get the Quote
Grab the related QQQ information and return 200 for success and the json or nothing. I highly suggest Postman to verify that you can get a valid response from the api first before you start trying to code. 
```python
def get_quote(symbols, is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    symbols_str = ','.join(symbols)
    url = f"{base_url}markets/quotes?symbols={symbols_str}"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    #print_rate_limits(response)
    if response.status_code == 200 and 'quotes' in response.json():
        quotes = response.json()['quotes']
        if 'quote' in quotes:
            return quotes['quote']
    return None
```

##The Heavy Lifter
Now we'll skip the easy stuff like grabbing options and expirations and talk about the heavy stuff. IV and Gamma/Vanna. THe gamma calculations are straight forward: 
Gamma * OI * 100 * Spot price ^2 = Gamma Exposure as seen here: https://perfiliev.co.uk/market-commentary/how-to-calculate-gamma-exposure-and-zero-gamma-level/

When we start dealing with Vanna Exposure though we have two problems. What is IV and What is Vanna?
To calculate IV you actually need to run it through the Black Scholes model and the easiest way to do this is to utilize the Vectorized_Implied_volatility module from py_vollib_vectorized library. 
```python
def calculate_iv(S, K, t, r, price, option_type):
    if option_type == 'call':
        flag = 'c'
    else:
        flag = 'p'

    # implied_volatility = vectorized_implied_volatility(
    #     price, S, K, t, r, [flag], q=0, model='black_scholes_merton', return_as='numpy'
    # )

    implied_volatility = vectorized_implied_volatility(
        price, S, K, t, r, [flag], q=0, model='black', return_as='numpy'
    )

    return implied_volatility[0]
```

Vanna is less straightforward as you need to calculate it with the new IV and then you can find the VEX afterwards with:
Vanna * OI * 100 * Spot

A lot of opinions with this calculation and so far this calculation is the only one I've seen that's not closed source that makes sense to track Vanna against Deltas. 
```python
def calculate_vanna(S, K, r, sigma, T):
    #d1 = (math.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
    d1 = (math.log(S / K) + (r - 0 + 0.5 * (sigma ** 2)) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    q = 0

    #vanna = (norm.pdf(d1) * d2) / sigma

    vanna = math.exp(-q * T) * norm.pdf(d1) * (d2 / sigma)

    return vanna
```

The math in the Vanna took some time to understand but with a little chatgpt understanding and realizing what each piece means we can identify the proper structure:

math.exp(-q * T) = Exponential Decay factor of dividends by time
norm.pdf(d1) = The probability density function of the standard normal distribution evaluated at d1.
d2 / sigma =  The adjustment factor for the volatility and the difference in the values of d1 and d2.

I hope some of these explanations and ideas help you make a deeper understanding of GEX/VEX.



