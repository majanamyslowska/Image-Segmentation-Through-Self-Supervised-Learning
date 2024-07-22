import os
import torch
from data_loader import pets_test
current_directory = os.getcwd()


def get_species_id():
    directory = os.getcwd()
    with open(directory + '/oxford_data/oxford-iiit-pet/annotations/test.txt', 'r') as l:

        get_species = lambda char: 1 if char.isupper() else 0
        
        species_id = []
        
        for line in l:
            species_id.append(get_species(line[0]))
        
        return species_id


species_id = get_species_id()
cats_test = []
dogs_test = []

for i, sample in enumerate(pets_test):
    if species_id[i] == 1:
        cats_test.append(sample)
    else:
        dogs_test.append(sample)

cats_test_loader = torch.utils.data.DataLoader(
        cats_test,
        batch_size=16,
        shuffle=True,
    )

dogs_test_loader = torch.utils.data.DataLoader(
        dogs_test,
        batch_size=16,
        shuffle=True,
    )