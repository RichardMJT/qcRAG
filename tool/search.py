from langchain_community.utilities import SerpAPIWrapper
import os
os.environ["SERPAPI_API_KEY"] = 'f094545df80c54955d8bf22ad488ba06a2feed3240ffa7fdc2896763376b898d'

params = {
    "engine": "google",
    "gl": "cn",
    "hl": "zh-CN",
}

def get_web_search_tool():    
    web_search_tool = SerpAPIWrapper(params=params)
    return web_search_tool