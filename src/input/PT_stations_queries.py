# Overpass API Endpoint
OVERPASS_URL = "https://overpass.private.coffee/api/interpreter"

QUERIES = {
    'metro': [
        """
        [out:json][timeout:300];
        area["boundary"="administrative"]["name"="Barcelona"]["admin_level"="8"]->.searchArea;
        relation(area.searchArea)["type"="route"]["route"~"subway|train|light_rail"]->.subwayRoutes;
        (.subwayRoutes;>;)->.subwayDetails;
        relation[public_transport=stop_area](bn.subwayDetails)->.stopAreas;
        (.stopAreas;>;)->.stopAreaDetails;
        node.stopAreaDetails["railway"="subway_entrance"]->.entrances;
        (.subwayRoutes; .subwayDetails; .stopAreas; .stopAreaDetails; .entrances;)->.all;
        .all out body;
        """,

        """ 
        [out:json][timeout:300];
        area["boundary"="administrative"]["name"="Barcelona"]["admin_level"="8"]->.searchArea;
        (
            relation(area.searchArea)["type"="route"]["route"~"subway|train|light_rail"];
            relation(area.searchArea)["public_transport"="stop_area"];
        )->.subwayRoutes;
        (.subwayRoutes;>;);
        out body;
        """
    ],
    
    'tram': [
        """
        [out:json][timeout:120];
        area["boundary"="administrative"]["name"="Barcelona"]["admin_level"="8"]->.searchArea;
        relation(area.searchArea)["type"="route"]["route"="tram"]->.tramRoutes;
        (.tramRoutes;>;);
        out body;
        """
    ],

    'bus': [
        """
        [out:json][timeout:900];
        area["boundary"="administrative"]["name"="Barcelona"]["admin_level"="8"]->.searchArea;
        relation(area.searchArea)["type"="route"]["route"="bus"]->.busRoutes;
        (.busRoutes;>;)->.allMembers;
        node.allMembers(area.searchArea)["highway"="bus_stop"];
        out body;
        """
    ]

}