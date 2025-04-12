from typing import Optional
import time
import requests
import json
import urllib.request
import ssl
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get the current weather at the given location using the WeatherStack API.

    Args:
        location: The location (city name).
        celsius: Whether to return the temperature in Celsius (default is False, which returns Fahrenheit).

    Returns:
        A string describing the current weather at the location.
    """
    api_key = "your_api_key"  # Replace with your API key from https://weatherstack.com/
    units = "m" if celsius else "f"  # 'm' for Celsius, 'f' for Fahrenheit

    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={location}&units={units}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()

        if data.get("error"):  # Check if there's an error in the response
            return f"Error: {data['error'].get('info', 'Unable to fetch weather data.')}"

        weather = data["current"]["weather_descriptions"][0]
        temp = data["current"]["temperature"]
        temp_unit = "°C" if celsius else "°F"

        return f"The current weather in {location} is {weather} with a temperature of {temp} {temp_unit}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Converts a specified amount from one currency to another using the exchangeratesapi.io API.
    Note: The free plan for this API uses EUR as the base currency.

    Args:
        amount: The amount of money to convert.
        from_currency: The currency code of the currency to convert from (e.g., 'USD').
        to_currency: The currency code of the currency to convert to (e.g., 'EUR').

    Returns:
        str: A string describing the converted amount in the target currency, or an error message if the conversion fails.
    """
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if not api_key:
        return "Error: EXCHANGERATE_API_KEY environment variable not set."

    # Ensure currency codes are uppercase
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    if from_currency == to_currency:
        return f"{amount} {from_currency} is equal to {amount} {to_currency}."

    url = f"https://api.exchangeratesapi.io/v1/latest?access_key={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        data = response.json()

        if not data.get("success"):
            error_info = data.get("error", {})
            error_msg = f"API Error Code {error_info.get('code')}: {error_info.get('info', 'Unknown API error')}"
            return f"Error fetching conversion data: {error_msg}"

        rates = data.get("rates")
        base_currency = data.get("base", "EUR") # API uses EUR as base

        if not rates:
            return "Error: Could not retrieve exchange rates from API response."

        # Get rates relative to the base currency (EUR)
        rate_from_eur = rates.get(from_currency)
        rate_to_eur = rates.get(to_currency)

        # Handle if base currency itself is requested
        if from_currency == base_currency:
            rate_from_eur = 1.0
        if to_currency == base_currency:
            rate_to_eur = 1.0

        if rate_from_eur is None:
            return f"Error: Unable to find exchange rate for {from_currency} relative to {base_currency}."
        if rate_to_eur is None:
            return f"Error: Unable to find exchange rate for {to_currency} relative to {base_currency}."

        # Perform the conversion via the base currency (EUR)
        # amount_in_eur = amount / rate_from_eur
        # converted_amount = amount_in_eur * rate_to_eur
        # Simplified:
        converted_amount = amount * (rate_to_eur / rate_from_eur)

        return f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}."

    except requests.exceptions.RequestException as e:
        # Handles connection errors, timeouts, etc.
        return f"Error fetching conversion data: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Failed to decode API response."
    except Exception as e:
        # Catch any other unexpected errors during processing
        return f"An unexpected error occurred during currency conversion: {str(e)}"

def get_time_in_timezone(location: str) -> str:
    """
    Fetches the current time for a given location using the World Time API.
    Args:
        location: The location for which to fetch the current time, formatted as 'Region/City'.
    Returns:
        str: A string indicating the current time in the specified location, or an error message if the request fails.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    url = f"http://worldtimeapi.org/api/timezone/{location}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        current_time = data["datetime"]

        return f"The current time in {location} is {current_time}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching time data: {str(e)}"

def search_google(query: str, location: Optional[str] = None, language: str = "en", country: str = "us") -> str:
    """
    Performs a Google search using Bright Data's SERP API and returns formatted results.
    Only returns organic search results without images.
    
    Args:
        query: The search query string
        location:  Location to localize search results (e.g. "Austin, Texas, United States") 
        language: Language code for results (default "en")
        country: Country code for results. MANDATORY.
        
    Returns:
        str: A markdown-formatted string containing only the organic search results
        
    Raises:
        Exception: If there is an error performing the search
    """
    try:
        # Disable SSL verification
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Configure proxy with authentication
        proxy_url = 'http://brd-customer-hl_10d3d7a9-zone-serp_api2:7edm2leyjmgg@brd.superproxy.io:33335'
        
        # Build URL opener with proxy
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({
                'http': proxy_url,
                'https': proxy_url
            })
        )
        
        # Construct search URL with parameters
        search_url = f'https://www.google.{country}/search?q={urllib.parse.quote(query)}'
        if language:
            search_url += f'&hl={language}'
        if country:
            search_url += f'&gl={country}'
        if location:
            search_url += f'&location={urllib.parse.quote(location)}'
            
        # Add brd_json parameter for JSON response format
        search_url += '&brd_json=1'
            
        # Make request
        response = opener.open(search_url)
        response_data = response.read().decode('utf-8')
        
        try:
            # Try to parse as JSON
            data = json.loads(response_data)
        except json.JSONDecodeError:
            return "Error: Unable to parse search results"
            
        # Format results as markdown - only organic results
        markdown = "### Google Search Results\n\n"
        
        # Add organic results
        if "organic" in data:
            for result in data["organic"]:
                # Create a clean result dictionary without image fields
                clean_result = {
                    "title": result.get("title", "No title"),
                    "link": result.get("link", "#"),
                    "description": result.get("description", "No description available"),
                    "display_link": result.get("display_link", result.get("link", "#")),
                    "rank": result.get("rank"),
                    "global_rank": result.get("global_rank")
                }
                
                markdown += f"#### [{clean_result['title']}]({clean_result['link']})\n"
                markdown += f"**URL:** {clean_result['display_link']}\n"
                markdown += f"**Description:** {clean_result['description']}\n\n"
                
                # Add any extensions/missing terms
                if "extensions" in result:
                    for ext in result["extensions"]:
                        if ext.get("type") == "missing":
                            markdown += f"*Missing term: {ext.get('text', '')}*\n"
                
                markdown += "---\n\n"
                
        time.sleep(5)
        return markdown if "organic" in data else "No results found."
        
    except Exception as e:
        return f"Error performing Google search: {str(e)}"

def format_search_results_to_markdown(search_results: dict) -> str:
    """Convert Tavily search results to markdown format
    
    Args:
        search_results (dict): The raw search results from Tavily API
        
    Returns:
        str: Formatted markdown string
    """
    markdown = "### Search Results\n\n"
    # Handle case where input is a string representation of dict
    if isinstance(search_results, str):
        import ast
        search_results = ast.literal_eval(search_results)
    
    if 'results' not in search_results:
        return "Error: Invalid search results format"
        
    for result in search_results['results']:
        # Add title with link
        markdown += f"#### [{result['title']}]({result['url']})\n\n"
        
        # Add content if available
        if result.get('content'):
            markdown += f"{result['content']}\n\n"
        
        # Add score if available
        if result.get('score'):
            markdown += f"*Relevance score: {result['score']}*\n\n"
        
        markdown += "---\n\n"  # Add separator between results
    return markdown.strip()

def read_webpage(query: str) -> str:
    """Read webpage content using Jina Reader API.
    
    Args:
        query: The URL of the webpage to read
        
    Returns:
        str: The extracted text content of the webpage if successful, 
             or an error message if the request fails
             
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request
    """
    try:
        jina_url = f'https://r.jina.ai/{query}'
        headers = {
            'Authorization': 'Bearer jina_2973abd594684126b8d3ca2efb04408aKbPL-FF1UedPb5EUwdb5XXUWo8OF'
        }
        response = requests.get(jina_url, headers=headers)
        return response.text
    except Exception as e:
        return f"Error reading webpage: {str(e)}"

def add(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Calculate the difference between two numbers."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Calculate the product of two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Calculate the division of two numbers. Handles division by zero."""
    if b == 0:
        return float('inf') # Or raise an error, depending on desired behavior
    return a / b
