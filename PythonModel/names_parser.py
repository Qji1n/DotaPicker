import json


def hero_id_to_index(hero_id):
    """Конвертировать hero_id в индекс для бинарного представления."""
    if hero_id == 126: return 0
    if hero_id == 128: return 24
    if hero_id == 129: return 115
    if hero_id == 135: return 116
    if hero_id == 136: return 117
    if hero_id == 137: return 118
    if hero_id == 138: return 122
    return hero_id


file_path = 'heroes.json'
name_id = dict()
id_name = dict()
with open(file_path) as file:
    data = json.load(file)
    for record in data:
        id = hero_id_to_index(data[record]["id"])
        localized_name = data[record]["localized_name"]
        name_id[localized_name] = id
        id_name[id] = localized_name

file_path = 'name_id.json'
with open(file_path, 'w') as file:
    json.dump(name_id, file)


file_path = 'id_name.json'
with open(file_path, 'w') as file:
    json.dump(id_name, file)
