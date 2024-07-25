import torch
import torch.nn.functional as F

class _ADClust_Autoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_size: int, act_fn=torch.nn.GELU):
        super(_ADClust_Autoencoder, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            act_fn(),
            torch.nn.Linear(512, 256),
            act_fn(),
            torch.nn.Linear(256, 128),
            act_fn(),
            torch.nn.Linear(128, embedding_size)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 128),
            act_fn(),
            torch.nn.Linear(128, 256),
            act_fn(),
            torch.nn.Linear(256, 512),
            act_fn(),
            torch.nn.Linear(512, input_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.decoder(embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction


    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn, scheduler=None):
        for _ in range(n_epochs):
            for batch, _ in trainloader:
                batch_data = batch.to(device)
                reconstruction = self.forward(batch_data)
                loss = loss_fn(reconstruction, batch_data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.train()
        for epoch in range(n_epochs):
            for batch_data in trainloader:
                
                # Assuming batch contains data and possibly labels
                if isinstance(batch, tuple) or isinstance(batch, list):
                    batch_data = batch[0]  # Extract data part
                else:
                    batch_data = batch

                batch_data = batch_data.to(device)

                # Add Gaussian noise to input for denoising autoencoder
                noisy_batch_data = batch_data + torch.randn_like(batch_data) * 0.1

                # Forward pass
                optimizer.zero_grad()
                output = self(noisy_batch_data)
                
                # Compute loss
                loss = loss_fn(output, batch_data)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Scheduler step (if provided)
            if scheduler:
                scheduler.step()

            # Print progress
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}')
