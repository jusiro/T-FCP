import torch
import numpy as np
import random

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

class Linear(torch.nn.Module):
    def __init__(self, F, C, W0=None, tau=1/100.0):
        super().__init__()
        self.F = F
        self.C = C
        self.W0 = W0
        self.N = None
        self.Nc = None

        # Init class weights.
        if W0 is None:
            self.W = torch.nn.Parameter(torch.nn.init.kaiming_normal_((1, C, F)))
        else:
            self.W = torch.nn.Parameter(self.W0.clone().unsqueeze(0))

        # Keep temperature scaling as in pre-training.
        if tau is not None:
            self.tau = torch.tensor(tau)
        else:
            self.tau = torch.tensor(1.0)
        self.tau.requires_grad = False

        # move to device.
        self.to(device)

    def forward(self, feats, norm_feats=True, norm_W=True):
        return self.forward_linear(feats, self.W, norm_feats=norm_feats, norm_W=norm_W)

    def forward_batched(self, feats, bs=50000, norm_feats=True, norm_W=True):
        preds, idx = [], 0
        while idx <= feats.size(0):                                                                               # Run over batches.
            preds.append(self.forward_linear(feats[idx:idx+bs, :], self.W, norm_feats=norm_feats, norm_W=norm_W)) # Forward batch.
            idx += bs                                                                                             # Update iterator.
        return torch.cat(preds, axis=1)

    def forward_linear(self, feats, W, norm_feats=True, norm_W=True):
        if norm_feats:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        if norm_W:
            W = W / W.norm(dim=-1, keepdim=True)
        return (feats.unsqueeze(0) @ W.transpose(1, 2)) / self.tau
    
    def forward_linear_batched(self, feats, W, bs=50000, norm_feats=True, norm_W=True):
        preds, idx = [], 0
        while idx <= feats.size(0):                                                                                     # Run over batches.
            preds.append(self.forward_linear(feats[idx:idx+bs, :], W, norm_feats=norm_feats, norm_W=norm_W))            # Forward batch.
            idx += bs                                                                                                   # Update iterator.
        return torch.cat(preds, axis=1).to(device)

    ## ----- LDA solver and online updater -----

    def fit_lda(self, X, Y, lambda_reg=10, lambda_text=1):
        
        W0 = self.W0

        # Constants, parameters
        self.N, self.F = X.shape
        self.Nc = np.bincount(Y)
        self.lambda_text = lambda_text
        self.lambda_reg = lambda_reg

        with torch.no_grad():
            
            # Initial mus
            self.mus = torch.cat([X[Y == i].mean(dim=0, keepdim=True) for i in range(self.C)])
            mus = self.mus + W0 * lambda_text

            # Inverse covariance matrix estimation.  
            Xc = torch.cat([X[Y == i] - W0[i,:].unsqueeze(0) for i in range(self.C)]) # Feature centering (class-wise)
            self.S = (Xc.T @ Xc)                                                      # Scatter matrix
            self.cov = self.S / (self.N)                                              # Covariance matrix
            trace = self.cov.trace()                                                  # Trace

            # Pseudo-invverse XtX+tr(Σ^)I
            cov_reg = self.cov + lambda_reg * torch.diag(self.cov) * torch.eye(self.F).to(device)
            self.cov_inv = torch.linalg.pinv(cov_reg)  

            # Correct mu to obtain class weights
            W = (self.cov_inv @ mus.t()).t()

            # Set linear weights
            self.W.data = W.to(device).unsqueeze(0).to(dtype) 
            self.cov_inv = self.cov_inv.to(dtype)
            self.mus = self.mus.to(dtype)

            
    def update_lda_from_inverse(self, x, Y):

        W0 = self.W0                                                  # (C, F)
        Nc = torch.as_tensor(self.Nc[Y], dtype=dtype, device=device)  # (C,)

        with torch.no_grad():

            # --- Update mean ---
            Nc_exp = Nc.unsqueeze(-1)                                    # (C, 1)
            mus_shifted = ((self.mus[Y,:] * Nc_exp) + x) / (Nc_exp + 1)  # (C, F)

            # Avoid repeat + loop → use broadcasting + indexing
            mus = self.mus.unsqueeze(0).expand(len(Y), -1, -1).clone()  # (C, C, F)
            idx = torch.arange(len(Y), device=device)
            mus[idx, Y, :] = mus_shifted.to(dtype)

            # Apply constraint to zero-shot prototypes
            mus = mus + W0.unsqueeze(0) * self.lambda_text

            # --- Update inverse covariance matrix ---

            # SO-LDA solution
            z = x - W0[Y]                                     # (C, F)
            z = z.unsqueeze(1)                                # (C, 1, F)

            cov_inv = self.cov_inv                            # (C, F, F) assumed batched

            # Sherman-Morrison-Woodbury for inverse covariance matrix updates.
            z_t = z.transpose(1, 2)                           # (C, F, 1)

            cov_z = cov_inv @ z_t                             # (C, F, 1)
            z_cov = z @ cov_inv                               # (C, 1, F)

            num = cov_z @ z_cov                               # (C, F, F)
            den = self.N + (z_cov @ z_t)                      # (C, 1, 1)

            cov_inv = ((self.N + 1) / self.N) * (cov_inv - num / den)

            # --- Compute LDA weights ---
            online_weights = (cov_inv @ mus.transpose(1, 2)).transpose(1,2)   # (C, C, F)

            return online_weights


    def update_lda_from_covariance(self, x, Y):
        
        W0 = self.W0                                                  # (C, F)
        Nc = torch.as_tensor(self.Nc[Y], dtype=dtype, device=device)  # (C,)

        with torch.no_grad():

            # --- Update mean ---
            Nc_exp = Nc.unsqueeze(-1)                                    # (C, 1)
            mus_shifted = ((self.mus[Y,:] * Nc_exp) + x) / (Nc_exp + 1)  # (C, F)

            # Avoid repeat + loop → use broadcasting + indexing
            mus = self.mus.unsqueeze(0).expand(len(Y), -1, -1).clone()  # (C, C, F)
            idx = torch.arange(len(Y), device=device)
            mus[idx, Y, :] = mus_shifted.to(dtype)

            # Apply constraint to zero-shot prototypes
            mus = mus + W0.unsqueeze(0) * self.lambda_text

            # --- Update inverse covariance matrix ---
            z = x - W0[Y]                                     # (C, F)
            zzt = z.unsqueeze(2) @ z.unsqueeze(1)             # (C, F, F)

            # Update covariance matrix
            cov = (self.S.unsqueeze(0) + zzt) / (self.N+1)

            # --- Regularization (recompute!) ---
            cov_reg = cov + self.lambda_reg * (cov.diagonal(dim1=1, dim2=2)).unsqueeze(-1) * torch.eye(self.F, device=device).unsqueeze(0)

            # --- Solve instead of inverse ---
            L = torch.linalg.cholesky(cov_reg)  # (C, F, F)

            online_weights = torch.cholesky_solve(
                mus.transpose(1, 2),  # (C, F, C)
                L
            ).transpose(1, 2)

            return online_weights


    def udate_lda_full(self, X, Y, x_online, y_candidates, lambda_text=1, lambda_reg=10):

        # Normalize features and zero-shot prototypes 
        X = X / X.norm(dim=-1, keepdim=True)
        x_online = x_online / x_online.norm(dim=-1, keepdim=True)
        W0 = self.W0 / self.W0.norm(dim=-1, keepdim=True)

        online_weights = torch.zeros((len(y_candidates), self.C, self.F))
        for i in range(len(y_candidates)):

            # Constants, parameters
            X_ext = torch.cat([X, x_online], dim=0)
            Y_ext = torch.cat([Y, y_candidates[i].unsqueeze(0)], dim=0)
            N, F = X_ext.shape
            Nc = np.bincount(Y_ext)
            with torch.no_grad():
                
                # Initial mus
                mus = torch.cat([X_ext[Y_ext == i].mean(dim=0, keepdim=True) for i in range(self.C)])

                # Apply constraint to zero-shot prototypes
                mus = mus + W0 * lambda_text

                # Inverse covariance matrix estimation.  
                Xc = torch.cat([X_ext[Y_ext == i] - W0[i,:].unsqueeze(0) for i in range(self.C)]) # Feature centering (class-wise)
                S = (Xc.T @ Xc)                                                                   # Scatter matrix
                cov = S / (N)                                                                     # Covariance matrix
                trace = cov.trace()                                                               # Trace

                # Pseudo-invverse XtX+tr(Σ^)I
                cov_reg = self.cov + lambda_reg * torch.diag(self.cov) * torch.eye(self.F).to(device)
                cov_inv = torch.linalg.pinv(cov_reg)  

                # Correct mu to obtain class weights
                W = (cov_inv @ mus.t()).t()

                online_weights[i,:,:] = W
        return online_weights.to(device)


    ## ----- Gradient Descent solver -----

    def fit_gd(self, features, labels, base_lr=0.1, epochs=300, bs=100000, x_online=None, Y_candidate=None):
        # Define constants
        N = features.shape[0]

        # Set initial weights and labels dimensioning
        if Y_candidate is not None:
            self.W = torch.nn.Parameter(self.W0.unsqueeze(0).expand(len(Y_candidate), -1, -1).clone())  # (C, C, F)
            features = torch.cat([features, x_online], dim=0) # (N+1, F)
            labels = labels.unsqueeze(0).to(device=device, dtype=torch.long).expand(len(Y_candidate), -1).clone()
            labels = torch.cat([labels, Y_candidate.to(device=device, dtype=torch.long).unsqueeze(-1)], dim=-1) # (C, N+1)
            N +=1
        else:
            labels = labels.unsqueeze(0).to(device=device, dtype=torch.long)

        # Set training optimizer
        optim = torch.optim.SGD(params=self.parameters(), lr=base_lr, weight_decay=0.0, momentum=0.9)

        # Set scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=epochs)

        # Set training indexes and shuffle
        indexes = torch.randperm(features.shape[0], device=features.device)
        features = features[indexes, :]
        labels = labels[:, indexes]

        # Fit Adapter
        for i_epoch in range(epochs):
            for i_step in range(max(1, N // bs)):

                # Select batch indexes
                init = int(i_step * bs)
                end = int((1 + i_step) * bs)

                # Forward
                logits = self.forward(features[init:end, :])
                
                # Adapter loss
                loss = torch.nn.functional.cross_entropy(logits.permute(1, 2, 0), labels[:, init:end].t())

                # Update model
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)

            # Update scheduler
            scheduler.step()

        if Y_candidate is not None:
            online_weights = self.W.data.clone()
            self.W = None
            return online_weights