##### HEALTH/CARE ###################################
DICT_HEALTH_CARE = {
    'amenity': ['pharmacy', 'clinic', 'doctors', 'hospital', 'dentist', 'therapist',
                'nursing_home', 'childcare', 'social_centre', 'social_facility'],
    'building': ['hospital'],
    'healthcare': ['pharmacy', 'clinic', 'podiatrist', 'doctor', 'dentist', 'nurse',
                   'physiotherapist', 'laboratory', 'hospital', 'alternative;physiotherapist',
                   'nutrition_counselling', 'therapist', 'blood_bank', 'rehabilitation',
                   'dialysis', 'audiologist', 'blood_donation', 'counselling']
}

##### CULTURE / TOURISM ###################################
DICT_CULTURE = {
    'historic': ['memorial', 'monument', 'castle', 'church', 'archaeological_site',
                 'wayside_cross', 'fort', 'ruins', 'archaeological_site;ruins', 
                 'heritage', 'aqueduct'],
    'amenity': [ 'library', 'arts_centre', 'exhibition_centre'],
    'building': ['library', 'museum', 'chapel', 'church', 'cathedral', 'temple', 'synagogue'],
    'tourism': ['museum', 'gallery', 'artwork']
}

DICT_TOURISM = {
    'tourism': ['hotel', 'hostel', 'guest_house', 'motel', 'museum',
                'theme_park', 'artwork', 'viewpoint', 'picnic_site',
                'winery', 'attraction', 'gallery', 'zoo', 'aquarium', 'information',
                'chalet'],
    'amenity': ['attraction', 'exhibition_centre', 'theatre', 'planetarium'],
    'building': ['hotel', 'museum']
}


##### TRANSPORT ###################################

##### ENTERTAINMENT/RECREATIONAL/RESTAURANTS ###################################
DICT_RECREATION = {
    'amenity': ['cinema', 'theatre', 'casino', 'nightclub', 'bar', 'pub', 
                'restaurant', 'fast_food', 'cafe', 'ice_cream', 'hookah_lounge',
                'karaoke_box', 'lounger', 'toy_library', 'food_court', 'nightclub', 
                'internet_cafe', 'hookah_lounge'],
    'building': ['restaurant'],
    'leisure': ['amusement_arcade', 'escape_game', 'bowling_alley', 'adult_gaming_centre',
        'tanning_salon', 'hackerspace', 'dance', 'bandstand', 'marina', 'esplai',
        'picnic_table', 'skill_game', 'amusement_arcade', 'flight_simulator',
        'sunbathing', 'swimming_area', 'nature_reserve'],
    'tourism': ['theme_park', 'zoo', 'aquarium', 'artwork', 'gallery', 'museum'],
}

##### SPORTS ###################################
DICT_SPORT = {
    'amenity': ['gym', 'track', 'dojo', 'sports_centre', 'stadium', 'sports_hall'],
    'leisure': ['sports_centre', 'swimming_pool', 'fitness_station', 'bowling_alley',
                'sports', 'climbing_wall', 'miniature_golf', 'climbing_wall', 
                'sport', 'horse_riding'],
    'sport': [
        'table_tennis', 'multi', 'swimming', 'fitness', 'yoga', 'gymnastics',
        'board_games', 'climbing', 'castells', 'skating', 'basketball',
        'badminton', 'teakwondo', 'martial_arts', 'surfing', 'skate', 'snowboard', 
        'fencing', 'fitness', 'bodybuilding', 'boxing', 'mixed_martial_arts',
        'brazilian_jiu_jitsu', 'muay_thai', 'kick_boxing', 'sailing', 'skateboard', 
        'running', 'soccer', 'karate', 'exercise', 'chess', 'rc_car',
        'table_soccer', 'muay_thai', 'orienteering', 'equestrian', 'shooting', 
        'pilates', 'padel', 'calisthenics', 'billiards', 'cycling', 
        'roller_skating', 'paintball', 'boules', 'aikido', 'kayaking',
        'billiard', 'snooker', 'beachvolleyball', 'dog_agility', 'tennis',
        'bmx', 'rugby_league', 'ice_skating', 'field_hockey', 'polo',
        'athletics', 'motor', 'long_jump', 'pole_vault', 'archery',
        'racquet', 'volleyball', 'futsal', 'handball', 'roller_hockey',
        'cricket', 'baseball', 'softball'
    ],
}

##### RETAIL/TRADE/ECONOMIC ###################################
DICT_ECONOMIC_RETAIL = {
    'amenity': ['marketplace', 'atm', 'bank', 'bureau_de_change', 'money_transfer'],
    'office': ['insurance', 'lawyer', 'estate_agent', 'financial', 'tax_advisor'],
    'shop': [
        'nan', 'doityourself', 'convenience', 'copyshop', 'computer',
        'supermarket', 'florist', 'massage', 'sports', 'bakery', 'butcher',
        'hairdresser', 'optician', 'confectionery', 'gift', 'jewelry', 'books',
        'electronics', 'chemist', 'stationery', 'bicycle', 'furniture',
        'greengrocer', 'clothes', 'laundry', 'shoes', 'yes', 'photo',
        'musical_instrument', 'kiosk', 'mobile_phone', 'art', 'bag',
        'car_repair', 'frozen_food', 'alcohol', 'mall', 'vacant', 'newsagent',
        'hardware', 'erotic', 'pet', 'medical_supply', 'car', 'seafood',
        'pastry', 'outdoor', 'garden_centre', 'medical', 'travel_agency',
        'motorbike rent and travels', 'accessories', 'variety_store',
        'training_centre', 'household_supply_store', 'lawyer', 'beauty',
        'dry_cleaning', 'motorcycle', 'music', 'deli', 'haberdashery', 'toys',
        'estate_agent', 'paint', 'car_parts', 'antiques', 'household_linen',
        'food', 'department_store', 'tobacco', 'watches', 'baby_goods',
        'copyshop;books', 'games', 'lottery', 'fabric', 'tailor', 'locksmith',
        'boutique', 'houseware', 'cosmetics', 'perfumery', 'pet_grooming',
        'motorcycle_repair', 'dog_wash', 'cannabis', 'herbalist', 'sewing',
        'wine', 'pasta', 'general', 'coffee', 'beverages', 'bed', 'ticket',
        'tea', 'bikes', 'rental', 'tyres', 'hearing_aids', 'video',
        'radiotechnics', 'tableware', 'interior_decoration', 'dairy',
        'agrarian', 'electrical', 'tattoo', 'health_food', 'yoga', 'anime',
        'second_hand', 'surf;skate;snowboard', 'carpet', 'candles', 'collector',
        'fishing', 'chocolate', 'bookmaker', 'charcutier;cheese;wine;butcher',
        'printing', 'head_shop', 'curtain', 'craft', 'bathroom_furnishing',
        'galician', 'printer_ink', 'model', 'cheese', 'nutrition_supplements',
        'psychic', 'leather', 'charity', 'religion', 'fashion_accessories',
        'money_lender', 'pottery', 'e-cigarette', 'appliance', 'video_games',
        'hookah', 'hairdresser_supply', 'party', 'frame', 'shoe_repair',
        'storage_rental', 'military_surplus', 'hifi', 'pawnbroker',
        'pyrotechnics', 'telecommunication', 'electric_supplies',
        'mobile_phone_accessories', 'hvac', 'vacuum_cleaner', 'lighting',
        'stairs', 'nuts', 'trade', 'watch_repair', 'hammock', 'kitchen',
        'ice_cream', 'bus_tickets', 'wool', 'delivery', 'tool_hire',
        'car;car_repair', 'bicycle_repair', 'farm', 'greengrocer;health_food',
        'wholesale', 'doors', 'maps', 'mobility_scooter', 'flooring', 'grocery',
        'repair', 'skate', 'confectionery;ice_cream', 'catalogue', 'knives',
        'scientific_equipment', 'weapons', 'orthopedics', 'electric_scooter',
        'printshop', 'camera', 'outpost', 'shisha', 'merchandise',
        'funeral_directors', 'aluminiom_carpentry'
    ]
}

DICT_INDUSTRIAL = {
    'landuse': ['industrial', 'depot', 'warehouse', 'quarry'],
    'building': ['industrial', 'warehouse', 'manufacture', 'factory']
}

##### PARKS/GREEN ###################################
DICT_GREEN_NATURE = {
    'landuse': ['park', 'forest', 'meadow', 'garden', 'village_green'],
    'leisure': ['park', 'garden', 'nature_reserve'],
    'natural': ['forest', 'beach', 'garden', 'wood', 'grassland', 'heath', 'shrubbery',
                'grassland', 'scrub', 'shrub', 'shrubbery']
}

##### CIVIC INSTITUTIONS ###################################
DICT_CIVIC = {
    'amenity': ['townhall', 'courthouse', 'police', 'prison', 'fire_station'],
    'building': ['government'],
    'office': ['government'],
}

##### WORSHIP ###################################
DICT_WORSHIP = {
    'amenity': ['place_of_worship'],
    'building': ['church', 'monastery', 'synagogue', 'cathedral', 'basilica'],
    'historic': ['church'],
    'landuse': ['religious']
}

##### EDUCATION ###################################
DICT_EDUCATION = {
    'amenity': ['school', 'college', 'university', 'language_school', 'music_school',
                'driving_school', 'beauty_school', 'training', 'prep_school', 
                'dancing_school'],
    'building': ['school', 'university', 'college'],
    'office': ['educational_institution']
}

##### SUPERPOIs ###################################
DICT_SUPERPOIS = {
    'leisure': ['stadium', 'sports_centre', 'sports_hall', 'concert_venue'],
    'tourism': ['theme_park', 'zoo', 'aquarium'],
    'natural': ['beach', 'coastline'],
    'landuse': ['park'],
    'building': ['stadium', 'sports_hall', 'concert_hall'],
    'amenity': ['events_venue', 'exhibition_centre', 'conference_centre'],
    'shop': ['mall', 'department_store'],
    'water': ['lake', 'river'],
    'waterway': ['river']
}



