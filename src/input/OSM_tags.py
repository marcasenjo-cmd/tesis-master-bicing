# https://wiki.openstreetmap.org/wiki/Map_Features
# https://wiki.openstreetmap.org/wiki/Key:building
# https://wiki.openstreetmap.org/wiki/Key:amenity
# https://wiki.openstreetmap.org/wiki/Key:tourism
# Building tags to filter residential buildings

RESIDENTIAL_BUILDING_TAGS = [
    'apartments', 'barracks', 'bungalow', 'cabin', 'detached', 'annexe', 
    'dormitory', 'farm', 'ger', 'hotel', 'house', 'houseboat', 'residential', 
    'semidetached_house', 'static_caravan', 'terrace', 'tree_house', 'trullo',
    'isolated_dwelling'
]


NON_RESIDENTIAL_BUILDING_TAGS = [
    # Comercial
    'commercial', 'industrial', 'kiosk', 'office', 'retail', 'supermarket', 'warehouse',

    # Religious buildings
    'cathedral', 'chapel', 'church', 'temple', 'kingdom_hall', 'monastery', 'mosque', 
    'presbytery', 'shrine', 'synagogue', 'temple', 'monastery', 'religious', 'cloister'

    # Civic / Infrastructure buildings
    'bakehouse', 'civic', 'college', 'fire_station', 'government', 'hospital', 
    'kindergarten', 'public', 'school', 'toilets', 'train_station', 'transportation', 
    'university', 'museum', 'annex', 'administrative', 'kindergarten', 'library',
    'office', 'public', 'school', 'shelter', 'sports', 'market', 'oficinas', 'palace',
    'bandstand', 'bridge', 'car_repair', 'casino', 'cloister', 'bell_tower', 'carport',
    'cinema',  'townhall', 'hall',

    # Agricultural 
    'barn', 'conservatory', 'cowshed', ' farm_auxiliary', 'greenhouse', 
    'slurry_tank', 'stable', 'sty', 

    # Sports buildings
    'grandstand', 'pavilion', 'riding_hall', 'sports_hall', 'sports_centre',
    'stadium',

    # Storage buildings
    'allotment_hall', 'boathouse', 'hangar', 'hut', 'shed', 'barn'
    
    # Automobile buildings
    'car_garage', 'garage', 'garages', 'parking',

    # Electricity / Technical buildings
    'digester', 'service', 'tech_cab', 'transformer_tower', 'water_tower', 
    'storage_tank', 'silo', 'manufacture', 'air_shaft',

    # Other buildings
    'beach_hut', 'bunker','castle', 'construction', 'container', 'guardhouse', 
    'military', 'outbuilding', 'pagoda', 'quonset_hut', 'roof', 'ruins', 'tent', 
    'tower', 'windmill', 'hotel', 'RE', 'Hotel_Acta_Splendid', 'industrial', 'c',
    'roof',  'terrace', 'demolished', 'proposed', 
    
    # Rare things
    # 'user defined', 
    # 'yes' # A lot of residential buildings are tagged as 'yes'.
    # Mostly those that the block of buildings are not divided in buildings.
    # However, some non-residential buildings are also tagged as 'yes'. Check manually
    # in the raw/osm_data/buildings/ folder to decide if they should be included or not.
    'yes;central_office', 'roof;yes', 'industrial;yes', 'yes;commercial', 'yes;university'
]

NON_RESIDENTIAL_AMENITY_TAGS = [
    'bicycle_parking', 'motorcycle_parking', 'parking_space', 'parking_entrance', 
    'bicycle_rental', 'shelter', 'research_institute', 'smoking_area', 'table', 
    'toilets', 'waste_disposal', 'recycling', 'garden', 'coworking_space', 
    'dog_toilet', 'drinking_water', 'bbq', 'public_building', 'atm', 'beauty_school',
    'bench', 'bicycle_rental;left_luggage', 'charging_station', 'childcare',
    'clock', 'compressed_air', 'dance_school', 'disused', 'dojo',
    'dressing_room', 'fixme', 'flight_attendant', 'grit_bin', 'grocery',
    'hookah_lounge', 'karaoke_box', 'kick-scooter_parking', 'letter_box',
    'loading_dock', 'locker', 'lounger', 'luggage_locker', 'office',
    'parcel_locker', 'pastries', 'photo_booth', 'piano', 'post_box', 'relay_box',
    'sailing_school', 'scooter_rental', 'shower', 'signs', 'stock_exchange',
    'surf_school', 'swingers club', 'tap', 'telephone', 'therapist', 'union',
    'vacuum_cleaner', 'vending_machine', 'warehouse', 'waste_basket',
    'water_point', 'watering_place', 'wifi',

    # Valencia
    'ticket_validator','Geldwechselstube', 'lavoir', 'meeting_point', 'parking_exit',
    'retirement_home', 'security_control', 
]


# Amenity tags to filter job locations
JOB_AMENITY_TAGS = [
    # Sustence
    'bar', 'bear_box', 'biergarten', 'cafe', 'canteen', 'fast_food', 'food_court', 
    'ice_cream', 'pub', 'restaurant', 'market',

    # Education
    'college', 'dancing_school', 'driver_training', 'driving_school', 'kindergarten', 
    'language_school', 'library', 'toy_library', 'music_school', 'prep_school', 
    'school', 'ski_school', 'training', 'university',

    # Transport
    'bicycle_repair_station', 'boat_rental', 
    'boat_sharing', 'boat_storage', 'bus_station', 'car_rental', 'car_sharing', 
    'car_wash', 'ferry_terminal', 'fuel', 'motorcycle_rental', 'parking',  
    'taxi', 'traffic_park', 'vehicle_inspection',  'weighbridge', 
    # 'compressed_air', 'grit_bin', 'vacuum_cleaner', 'vehicle_ramp','parking_entrance', 
    # 'ticket_validator', 'bicycle_parking', 'charging_station', 'motorcycle_parking',
    # 'parking_space', 'bicycle_rental', 

    # Financial
    'payment_terminal', 'bank', 'bureau_de_change', 'money_transfer', 
    'payment_centre', # 'atm',

    # Healthcare
    'baby_hatch', 'clinic', 'dentist', 'doctors', 'hospital', 'nursing_home', 
    'pharmacy', 'social_facility', 'veterinary',

    # Entertainment, Art and Culture
    'arts_centre', 'brothel', 'casino', 'cinema', 'community_centre', 
    'conference_centre', 'events_venue', 'exhibition_centre', 
    'fountain', 'gambling', 'love_hotel', 'music_venue', 'nightclub', 
    'planetarium', 'public_bookcase', 'social_centre', 'stage', 
    'stripclub', 'studio', 'swingerclub', 'theatre', 'exhibition_hall',

    # Public Service
    'courthouse', 'fire_station', 'police', 'post_depot', 'post_office', 
    'prison', 'ranger_station', 'townhall', # 'post_box',

    # # Facilities
    # 'bbq', 'bench', 'dog_toilet', 'dressing_room', 'drinking_water', 'give_box', 'mailroom',
    # 'parcel_locker', 'shelter', 'shower', 'toilets', 'trolley_bay', 
    # 'water_point', 'watering_place',

    # Waste Management
    'sanitary_dump_station', 'waste_transfer_station', #'recycling', 'waste_disposal', 'waste_basket', 

    # Other
    'animal_boarding', 'animal_breeding', 'animal_shelter', 'animal_training', 'baking_oven', 
    'crematorium', 'dive_centre', 'funeral_hall', 'grave_yard', 'hunting_stand', 'internet_cafe', 
    'kitchen', 'kneipp_water_cure',  'marketplace', 
    'monastery', 'mortuary', 'place_of_mourning', 'place_of_worship', 
    'public_bath', 'refugee_site', #'clock', 'lounger', 'photo_booth', 'vending_machine'

    # Rare things
    # 'user defined'
]

JOB_TOURISM_TAGS = [
    'aquarium', 'attraction', 'gallery', 'hostel', 'information', 'motel',
    'theme_park', 'zoo', 'winery', 'viewpoint', 'artwork', 'museum', 'yes',

    # 'alpine_hut', 'apartment', 'artwork',  'camp_pitch',
    # 'camp_site', 'cavan_site', 'chalet',  'guest_house',  'museum', 'picnic_site','trail_riding_station', 'viewpoint',
    # 'wilderness_hut', 'yes'
]

JOB_LANDUSE_TAGS = [
    # Developed land
    'construction', 'industrial',
    # commercial, retail, education, and residential are not chosen because 
    # they are covered by buildings and amenity
    # military is not explicit enough if it is a home or work


    # Rural and agricultural land
    'farmland', 'farmyard', 'orchard', 'greenhouse_horticulture', 'plant_nursery'
]

# Study related tags
STUDY_RELATED_TAGS = [
    # From non_residential_building_tags
    'college', 'kindergarten', 'school', 'university',

    # From job_amenity_tags
    'college', 'dancing_school', 'driver_training', 'driving_school', 'kindergarten',
    'language_school', 'music_school', 'prep_school', 'school', 'training', 'university'
]