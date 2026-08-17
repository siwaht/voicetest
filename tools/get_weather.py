from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Use this whenever the user asks about the weather, temperature, or
    conditions somewhere. Pass just the city name, for example "Portland".

    Returns a one-line summary of temperature and conditions.
    """
    # Placeholder: returns a fixed response rather than calling a weather API.
    return f"The weather in {city} is 18 degrees celsius and rainy"
