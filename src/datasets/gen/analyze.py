"""
Yamnet is subset of AudioSet
SemanticHearing is subset of AudioSet
SemanticHearing is almost subset of Yamnet "without music(Melody in AudioSet)"
"""
import json
import csv
import yaml

def main():
    # Load ontology.json
    with open("ontology.json", "r") as f:
        ontology = json.load(f)

    # Load yamnet_class_map.csv
    yamnet_class_list = []
    with open("yamnet_class_map.csv", "r") as f:
        yamnet_class_csv = csv.reader(f)
        # convert to dictionary
        for idx, row in enumerate(yamnet_class_csv):
            if idx == 0:
                continue
            yamnet_class_list.append({
                'id': row[1],
                'name': row[2]
            })

    # The class we choose(Classes.yaml) are subset of AudioSet
    semhl_classes_yaml = []
    with open("Classes.yaml", "r") as f:
        semhl_classes_yaml = yaml.safe_load(f)

    semhl_classes_list = []
    for class_name, class_name_audioset in semhl_classes_yaml.items():
        print(f"Class name: {class_name} AudioSet name: {class_name_audioset[0]}")
        # find the id in audioset
        id = None
        for ontology_class in ontology:
            if class_name_audioset[0] == ontology_class['name']:
                id = ontology_class['id']
                break
        semhl_classes_list.append({
            'id': id,
            'name': class_name_audioset[0],
            'semhl_name': class_name
        })

    # Print the number of classes in the ontology
    print(len(ontology))
    print(ontology[0])

    # Print the number of classes in the yamnet_class_map
    print(len(yamnet_class_list))
    print(yamnet_class_list[0])

    # Print the number of classes in the semhl_classes
    print(len(semhl_classes_list))
    print(semhl_classes_list[0])

    found_classes = []
    for yamnet_class in yamnet_class_list:
        id = yamnet_class['id']
        name = yamnet_class['name']
        for ontology_class in ontology:
            if id in ontology_class['child_ids'] or id == ontology_class['id']:
                found_classes.append(name)
                break

    # Print the number of classes in the found_classes
    print(len(found_classes))
    if len(found_classes) == len(yamnet_class_list):
        print("YAMNet is subset of AudioSet")
    else:
        print("YAMNet is not subset of AudioSet")

    found_semhl_classes = []
    not_found_semhl_classes = []
    for semhl_class in semhl_classes_list:
        is_found = False
        id = semhl_class['id']
        name = semhl_class['name']
        for yamnet_class in yamnet_class_list:
            if id == yamnet_class['id']:
                found_semhl_classes.append(semhl_class['semhl_name'])
                is_found = True
                break

        if not is_found:
            not_found_semhl_classes.append(semhl_class['semhl_name'])

    # Print the number of classes in the found_semhl_classes
    print(len(found_semhl_classes))
    print(found_semhl_classes)

    # Print the number of classes in the not_found_semhl_classes
    print(len(not_found_semhl_classes))
    print(not_found_semhl_classes)

if __name__ == "__main__":
    main()