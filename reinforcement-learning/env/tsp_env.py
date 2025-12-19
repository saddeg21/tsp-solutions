import torch
from torch.utils.data import Dataset

class TSPDataset(Dataset):
    """
    @num_samples : How many TSP problem in set
    @num_cities : How many cities in each TSP problem
    """
    def __init__(self,num_sample=None, num_cities=None, file_path=None):
        if file_path is not None:
            self.coordinates = torch.load(file_path)
            print(f"TSP dataset loaded: {file_path}")
        elif num_sample is not None and num_cities is not None:
            # random creation
            self.coordinates = self.generate_data(num_sample, num_cities)
            print(f"TSP veriseti oluşturuldu: {num_sample} sample, {num_cities} city")      
        else:
            raise ValueError("Either file_path or both num_sample and num_cities must be provided.")
        

    def __len__(self):
        return (self.coordinates).size(0)
    
    def __getitem__(self, idx):
        return self.coordinates[idx]

    def generate_data(self, num_samples, num_cities):
        coords = torch.rand(num_samples, num_cities, 2)
        self.coordinates = coords
        return coords

    

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


