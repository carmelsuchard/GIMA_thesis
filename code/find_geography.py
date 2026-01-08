import requests
import rdflib
from rdflib.namespace import FOAF, RDF, RDFS, XSD, Namespace
from SPARQLWrapper import SPARQLWrapper, JSON
from config import TOKEN

def find_in_dbpedia(spatial):
    sparql = SPARQLWrapper(
    "https://dbpedia.org/sparql"
    )
    sparql.setReturnFormat(JSON)
    query = r"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX dbo: <http://dbpedia.org/ontology/>

        SELECT DISTINCT ?spatial ?label
        WHERE {
            ?spatial rdfs:label "spatial_tag"@en.
            {
                ?spatial rdf:type geo:SpatialThing;
                rdfs:label ?label.

            }
              UNION
            {
                ?spatial rdf:type dbo:Place;
                rdfs:label ?label.
            }
            }
        LIMIT 1
        """
    query = query.replace("spatial_tag", spatial)

    sparql.setQuery(
        query
    )
    
    try:
        ret = sparql.queryAndConvert()["results"]["bindings"][0]
        label, dbpedia_uri = ret["label"]["value"], ret["spatial"]["value"]
        return label, dbpedia_uri
    
    except Exception as e:
        return None, None


def clean_name_find_in_dbpedia(spatial):
    # Remove compass directions and locations within names
    directions = ["zuidelijke", "noordelijk", "oostelijk", "westelijk",
                  "zuid-West", "zuid-oost", "noord-west", "noord-oost", 
                  "zuid", "west", "oost", "noord",
                 "middengebied"]

    clean_spatial = spatial
    # Remove any instances of the directions from the name
    clean_spatial = clean_spatial.lower()
    for direction in directions:
        clean_spatial = clean_spatial.replace(direction, "")
    
    clean_spatial = clean_spatial.replace("-", "").replace("het ", "").replace("de ", "").title().strip().title()
    label, dbpedia_uri = find_in_dbpedia(clean_spatial)

    if dbpedia_uri:
        return label, dbpedia_uri
    else:
        return None, None
    
# def find_in_kadaster(spatial):
#     pass
    
def find_in_wikidata(spatial):
    ENDPOINT= "https://www.wikidata.org/w/rest.php/wikibase/v0/search/items"
    params = {"q": spatial,
              "limit": 10,
              "language": "en"
    }
    headers = {"Accept": "application/json",
               "User-Agent": "MyThesisScript/1.0 (Carmel Suchard; carmelsuchard@gmail.com)"}
    result = requests.get(ENDPOINT, headers=headers, params=params).json()
    print(result)

def find_in_geonames(spatial):    
    ENDPOINT= "http://api.geonames.org/searchJSON"
    # # http://api.geonames.org/search?q=london&maxRows=10&username=demo
    # http://api.geonames.org/searchJSON?q=london&maxRows=10&username=demo
    
    params = {"name_equals": {spatial},
              "name": {spatial},
              "maxRows": 10,
              "username": "robert",
              "searchlang": "nl",
              "name_equals": spatial,
              "isNameRequired": "true",
              "lang": "en"
              }
    headers = {"Accept": "application/json"}
    english_name = requests.get(ENDPOINT, headers=headers, params=params).json()["geonames"][0]["name"]
    
    label, dbpedia_uri = find_in_dbpedia(english_name)

    if dbpedia_uri:
        return label, dbpedia_uri
    else:
        return None, None
    


if __name__ == "__main__":
    # result = find_in_dbpedia("Overijssel")
    # result = clean_name_find_in_dbpedia("Hollendoorn")
    # result = find_in_geonames("Frankrijk")
    result = find_in_wikidata("Stedendriehoek")
    print(result)
    