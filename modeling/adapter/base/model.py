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
            self.W = torch.nn.Parameter(torch.nn.init.kaiming_normal_((C, F)))
        else:
            self.W = torch.nn.Parameter(self.W0.clone())

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
        return (feats @ W.t()) / self.tau

   ## ----- LDA solver -----

    def fit_lda(self, X, Y, lambda_reg=10, lambda_text=1):

        # Set features and l2-normalize
        W0 = self.W0

        # Constants
        self.N, self.F = X.shape
        self.Nc = np.bincount(Y)

        with torch.no_grad():
            
            # Initial mus and constraint to zero-shot prototypes
            self.mus = torch.cat([X[Y == i].mean(dim=0, keepdim=True) for i in range(self.C)])
            mus = self.mus + W0 * lambda_text

            # Inverse covariance matrix estimation.  
            Xc = torch.cat([X[Y == i] - W0[i,:].unsqueeze(0) for i in range(self.C)]) # Feature centering (class-wise)
            S = (Xc.T @ Xc)                                                           # Scatter matrix
            cov = S / (self.N)                                                        # Covariance matrix
            trace = cov.trace()                                                       # Trace

            # Pseudo-invverse XtX+tr(Σ^)I
            cov_reg = cov + lambda_reg * torch.diag(cov) * torch.eye(self.F).to(device)
            cov_inv = torch.linalg.pinv(cov_reg)  

            # Correct mu to obtain class weights
            W = (cov_inv @ self.mus.t()).t()

            # Set linear weights
            self.W.data = W.to(device).to(dtype)


   ## ----- Gradient Descent solver -----

    def fit_gd(self, features, labels, base_lr=0.1, epochs=300, bs=100000):

        # Define constants
        N = features.shape[0]

        # Set training optimizer
        optim = torch.optim.SGD(params=self.parameters(), lr=base_lr, weight_decay=0.0, momentum=0.9)

        # Set scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=epochs)

        # Fit Adapter
        for i_epoch in range(epochs):
            # Set training indexes
            indexes = np.arange(0, features.shape[0])
            random.shuffle(indexes)
            for i_step in range(max(1, N // bs)):

                # Select batch indexes
                init = int(i_step * bs)
                end = int((1 + i_step) * bs)

                # Forward
                logits = self.forward(features[indexes[init:end], :])

                # Adapter loss
                loss = torch.nn.functional.cross_entropy(logits, labels[indexes[init:end]])

                # Update model
                loss.backward()
                optim.step()
                optim.zero_grad()

            # Update scheduler
            scheduler.step()


   ## ----- Transductive Information Maximization solver -----

    def fit_tim(self, features_cal, features_test, label_marginal, alpha=0.1, base_lr=0.1, epochs=300, bs=100000):

        features = torch.cat([features_cal, features_test], dim=0)
        marginal = torch.tensor(label_marginal, device=device)

        # Define constants
        N = features.shape[0]

        # Set training optimizer
        optim = torch.optim.SGD(params=self.parameters(), lr=base_lr, weight_decay=0.0, momentum=0.9)

        # Set scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=epochs)

        # Fit Adapter
        for i_epoch in range(epochs):
            # Set training indexes
            indexes = np.arange(0, features.shape[0])
            random.shuffle(indexes)
            for i_step in range(max(1, N // bs)):

                # Select batch indexes
                init = int(i_step * bs)
                end = int((1 + i_step) * bs)

                # Forward
                logits = self.forward(features[indexes[init:end], :])
                pik = torch.softmax(logits, -1)
                pk = torch.mean(pik, dim=0)

                # Sample shannon entropy
                H_yx = - torch.mean(torch.sum(pik * torch.log(pik + 1e-3), -1))

                # Label marginal entropy
                Lmarg = torch.sum(marginal.clone() * torch.log((marginal.clone() / pk) + 1e-3))

                # Overall loss
                loss = Lmarg + alpha * H_yx

                # Update model
                loss.backward()
                optim.step()
                optim.zero_grad()

            # Update scheduler
            scheduler.step()


   ## ----- Transductive Optimal Transport Solver -----

    def postprocess_ot(self, logits_cal, logits_test, label_marginal):

        def distributed_sinkhorn(similarities, epsilon=0.8, num_iters=3, r=None, c=None):
            similarities = similarities.to(device)
            Q = torch.exp(similarities / epsilon).t()  # Q is K-by-B for consistency with notations
            K, B = Q.shape  # Number of clusters and batch size

            if r is None:
                r = torch.ones(K).to(similarities.device) / K
            if c is None:
                c = torch.ones(B).to(similarities.device) / B

            # make the matrix sums to 1
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            for it in range(num_iters):
                # normalize each row: total weight per prototype must be 1/K
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_of_rows
                Q *= r.unsqueeze(1)

                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q *= c.unsqueeze(0)

            Q *= B  # the columns must sum to 1 so that Q is an assignment
            Q = Q.t()

            return Q

        # Compute soft code pseudo-labels via optimal transport
        soft_code = distributed_sinkhorn(torch.cat([logits_cal, logits_test], dim=0),
                                         r=torch.tensor(label_marginal, device=device))
        
        # Retrieve calibration and test logits
        codes_cal, codes_test = soft_code[:logits_cal.shape[0],:], soft_code[logits_cal.shape[0]:,:]
        return codes_cal, codes_test
