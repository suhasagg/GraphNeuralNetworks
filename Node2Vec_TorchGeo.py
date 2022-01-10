" Node2Vec model using torch geometric with Cora"
import torch
from torch_geometric.datasets import Planetoid # The citation network datasets “Cora”, “CiteSeer” and “PubMed” 
from torch_geometric.nn import Node2Vec # Import Node2Vec Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

" **************** IMPORT DATA ********************"
path = "C:/Users/Suhas/Desktop"  # Directory to download dataset
dataset = Planetoid(path, "Cora") # Download the dataset
data = dataset[0] # Tensor representation of the Cora-Planetoid data
print('Coda: ', data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

" **************** CONSTRUCT THE MODEL  ********************"
Node2Vec_model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = Node2Vec_model.loader(batch_size=128, shuffle=True, num_workers=4) # For batch training
optimizer = torch.optim.SparseAdam(list(Node2Vec_model.parameters()), #List of parameters
                                   lr = 0.01 # Learning Rate
                                   )

" **************** TRAIN FUNCTION ********************"
def train():
    Node2Vec_model.train() # Set training as true for the model
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad() # reset of gradient of all variables
        loss = Node2Vec_model.loss(pos_rw , neg_rw)
        loss.backward()
        optimizer.step()
        total_loss =+ loss.item()
    return total_loss / len(loader)      


" **************** GET EMBEDDING  ********************"

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    
" **************** PLOT 2D OF EMBEDDED REP   ********************"
@torch.no_grad() # Deactivate autograd functionality
def plot_point(colors):
    Node2Vec_model.eval() # Evaluate the model based on the trained parameters
    z = Node2Vec_model(torch.arange(data.num_nodes, device=device)) # Embedding rep
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()
    plt.figure()
    for i in range(dataset.num_classes):
        plt.scatter(z[y==i,0],z[y==i,1],s=20,color=colors[i])
    plt.axis('off')
    plt.show()
colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
plot_point(colors)

" **************** NODE CLASSIFICATION ********************"
def test():
    Node2Vec_model.eval() # Evaluate the model based on the trained parameters
    z = Node2Vec_model() # Evaluate the model based on the trained parameters
    acc = Node2Vec_model.test(z[data.train_mask] ,data.y[data.train_mask],
                              z[data.test_mask],data.y[data.test_mask],
                              max_iter=150)
    return acc

print('Accuracy:', test())
