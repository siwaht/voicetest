from langchain.tools import tool


@tool
def get_weather(city:str):
    """"get weather of a city"""
    return f"The weather of  {city} is 18 degrees celcius and rainy"