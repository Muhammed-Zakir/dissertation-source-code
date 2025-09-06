class BatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that maintains a fixed ratio of labeled to unlabeled samples.
    """
    def __init__(self, dataset, batch_size, unlabeled_ratio, num_iterations=None):
        """
        Initialize the batch sampler.
        
        Args:
            dataset: FixMatch dataset with labeled and unlabeled data
            batch_size: Total samples per batch
            unlabeled_ratio: How many unlabeled samples per labeled sample
            num_iterations: Number of batches per epoch (auto-calculated if None)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.unlabeled_ratio = unlabeled_ratio

        # Split batch size between labeled and unlabeled samples
        self.num_labeled_per_batch = max(1, int(batch_size / (1 + unlabeled_ratio)))
        self.num_unlabeled_per_batch = self.num_labeled_per_batch * unlabeled_ratio

        # Update actual batch size after calculation
        self.batch_size = self.num_labeled_per_batch + self.num_unlabeled_per_batch

        # Create index lists for labeled and unlabeled data
        self.labeled_indices = list(range(self.dataset.total_labeled))
        self.unlabeled_indices = list(range(self.dataset.total_labeled, self.dataset.total_labeled + self.dataset.total_unlabeled))

        # Determine batches per epoch
        if num_iterations is None:
            self.num_batches_per_epoch = max(1, len(self.labeled_indices) // self.num_labeled_per_batch)
            
            max_labeled_batches = len(self.labeled_indices) // self.num_labeled_per_batch if self.num_labeled_per_batch > 0 else float('inf')
            max_unlabeled_batches = len(self.unlabeled_indices) // self.num_unlabeled_per_batch if self.num_unlabeled_per_batch > 0 else float('inf')
            
            self.num_batches_per_epoch = min(self.num_batches_per_epoch, int(min(max_labeled_batches, max_unlabeled_batches)))
        else:
            self.num_batches_per_epoch = num_iterations

        # Check if enough data
        if self.num_labeled_per_batch > 0 and len(self.labeled_indices) < self.num_labeled_per_batch:
            raise ValueError(f"Not enough labeled samples ({len(self.labeled_indices)}) for batch size {self.batch_size} with ratio {unlabeled_ratio}. Need at least {self.num_labeled_per_batch} labeled samples per batch.")
        
        if self.num_unlabeled_per_batch > 0 and len(self.unlabeled_indices) < self.num_unlabeled_per_batch:
            print(f"Warning: Not enough unlabeled samples ({len(self.unlabeled_indices)}) for batch size {self.batch_size} with ratio {unlabeled_ratio}. Need at least {self.num_unlabeled_per_batch} unlabeled samples per batch. Some batches might have fewer unlabeled samples.")

        print(f"BatchSampler configured:")
        print(f"  Batch Size: {self.batch_size} ({self.num_labeled_per_batch} labeled, {self.num_unlabeled_per_batch} unlabeled)")
        print(f"  Iterations per epoch: {self.num_batches_per_epoch}")

    def __iter__(self):
        """Generate batches of mixed labeled/unlabeled indices."""
        # Shuffle both sets for variety
        random.shuffle(self.labeled_indices)
        random.shuffle(self.unlabeled_indices)

        # Create iterators to pull
        labeled_iter = iter(self.labeled_indices)
        unlabeled_iter = iter(self.unlabeled_indices)

        for _ in range(self.num_batches_per_epoch):
            batch_indices = []
            try:
                # Add labeled samples to batch
                for _ in range(self.num_labeled_per_batch):
                    batch_indices.append(next(labeled_iter))

                # Add unlabeled samples to batch
                for _ in range(self.num_unlabeled_per_batch):
                    batch_indices.append(next(unlabeled_iter))

            except StopIteration:
                # Ran out of data before finishing all planned batches
                print("Warning: Ran out of data before completing all batches in an epoch. Consider increasing dataset size or adjusting batch/iteration counts.")
                break

            # Mix labeled and unlabeled samples within the batch
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        """Number of batches per epoch."""
        return self.num_batches_per_epoch