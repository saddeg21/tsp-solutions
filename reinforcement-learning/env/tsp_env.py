import torch
from torch.utils.data import Dataset

class TSPDataset(Dataset):
    """
    @num_samples : How many TSP problem in set
    @num_cities : How many cities in each TSP problem
    """
    def __init__(self,num_sample, num_cities):
        self.num_samples = num_sample
        self.num_cities = num_cities
        self.data = []
        self.generate_data()
        

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

    def generate_data(self):
        """
        for i in range(self.num_samples):
            cities = torch.rand(self.num_cities, 2)
            self.data.append(cities)
        FASTER WAY
        """
        self.data = torch.rand(self.num_samples, self.num_cities, 2)
    

def calculate_tour_length(tour, coordinates):
    """
    tour : (batch_size, num_cities) LongTensor
    coordinates : (batch_size, num_cities, 2) FloatTensor
    return : (batch_size,) FloatTensor
    """
    batch_size, num_cities = tour.size()
    
    # Gather the coordinates according to the tour
    idx = tour.unsqueeze(-1).expand(-1, -1, 2)  # (batch_size, num_cities, 2)
    ordered_coords = torch.gather(coordinates, 1, idx)  # (batch_size, num_cities, 2)
    
    # Calculate the distance between consecutive cities
    shifted_coords = ordered_coords.roll(-1, dims=1)  # Shift to get the next city
    segment_lengths = torch.norm(ordered_coords - shifted_coords, dim=-1)  # (batch_size, num_cities)
    
    # Sum up the lengths to get the total tour length
    tour_lengths = segment_lengths.sum(dim=1)  # (batch_size,)
    
    return tour_lengths


