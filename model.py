import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# Framework For Collaborative Attention Filtering -> Recommendation

class AttentionCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout_rate = 0.1):
        super(AttentionCollaborativeFiltering, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.attention_layer = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = 1, dropout = dropout_rate)

        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, item_ids):
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        combined = torch.cat((user_embedded.unsqueeze(1), item_embedded.unsqueeze(1)), dim = 1)
        attn_output, _ = self.attention_layer(combined, combined, combined)
        attn_sum = torch.sum(attn_output, dim = 1)
        ratings = self.output_layer(attn_sum).squeeze(1)
        
        return ratings
    
if __name__ == "__main__":
    num_users = 100
    num_items = 1000
    embedding_dim = 32

    user_ids = torch.randint(0, num_users, (1000,))
    item_ids = torch.randint(0, num_items, (1000,))
    ratings = torch.rand(1000)

    model = AttentionCollaborativeFiltering(num_users, num_items, embedding_dim)

    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predicted_ratings = model(user_ids, item_ids)
        loss = criterion(predicted_ratings, ratings)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    model.eval()  

    user_id = torch.tensor([0])  
    watched_item_ids = torch.tensor([10, 20, 30])  

    item_scores = {}
    for item_id in range(num_items):
        if item_id not in watched_item_ids:
            predicted_rating = model(user_id, torch.tensor([item_id]))
            item_scores[item_id] = predicted_rating.item()

    top_n_recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse = True)[:10]

    print("Top Recommendations:")
    for item_id, score in top_n_recommendations:
        print(f"Item ID: {item_id}, Predicted Score: {score}")
