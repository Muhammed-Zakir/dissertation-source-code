class FixMatchBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler for FixMatch that maintains a fixed ratio of labeled to unlabeled samples.
    
    Each batch will contain a mix of labeled and unlabeled data according to the specified ratio,
    which is essential for the FixMatch semi-supervised learning algorithm.
    """
    
    def __init__(self, dataset, batch_size, unlabeled_ratio, num_iterations=None):
        """
        Initialize the batch sampler.
        
        Args:
            dataset: FixMatch dataset containing both labeled and unlabeled data
            batch_size: Total samples per batch
            unlabeled_ratio: How many unlabeled samples per labeled sample (e.g., 7 means 7:1 ratio)
            num_iterations: Optional number of batches per epoch (auto-calculated if None)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.unlabeled_ratio = unlabeled_ratio

        # Figure out how to split each batch between labeled and unlabeled data
        self.num_labeled_per_batch = max(1, int(batch_size / (1 + unlabeled_ratio)))
        self.num_unlabeled_per_batch = self.num_labeled_per_batch * unlabeled_ratio

        # Update actual batch size in case of rounding
        self.batch_size = self.num_labeled_per_batch + self.num_unlabeled_per_batch

        # Split dataset indices by type
        self.labeled_indices = list(range(self.dataset.total_labeled))
        self.unlabeled_indices = list(range(
            self.dataset.total_labeled, 
            self.dataset.total_labeled + self.dataset.total_unlabeled
        ))

        # Determine how many batches we'll create per epoch
        if num_iterations is None:
            # Default: see each labeled sample roughly once per epoch
            self.num_batches_per_epoch = max(1, len(self.labeled_indices) // self.num_labeled_per_batch)
            
            # Make sure we don't exceed what's possible with available data
            max_labeled_batches = len(self.labeled_indices) // self.num_labeled_per_batch if self.num_labeled_per_batch > 0 else float('inf')
            max_unlabeled_batches = len(self.unlabeled_indices) // self.num_unlabeled_per_batch if self.num_unlabeled_per_batch > 0 else float('inf')
            
            self.num_batches_per_epoch = min(self.num_batches_per_epoch, int(min(max_labeled_batches, max_unlabeled_batches)))
        else:
            self.num_batches_per_epoch = num_iterations

        # Sanity checks - make sure have enough data
        if self.num_labeled_per_batch > len(self.labeled_indices):
            raise ValueError(
                f"Not enough labeled samples! Need {self.num_labeled_per_batch} per batch "
                f"but only have {len(self.labeled_indices)} total."
            )
            
        if self.num_unlabeled_per_batch > len(self.unlabeled_indices):
            print(
                f"Warning: Only {len(self.unlabeled_indices)} unlabeled samples available, "
                f"but need {self.num_unlabeled_per_batch} per batch. Some batches may be smaller."
            )

        # Print config summary
        print(f"FixMatch sampler ready:")
        print(f"  {self.batch_size} samples per batch ({self.num_labeled_per_batch} labeled + {self.num_unlabeled_per_batch} unlabeled)")
        print(f"  {self.num_batches_per_epoch} batches per epoch")

    def __iter__(self):
        """Generate batches of mixed labeled/unlabeled indices."""
        # Shuffle both sets at the start of each epoch for variety
        random.shuffle(self.labeled_indices)
        random.shuffle(self.unlabeled_indices)

        # Create iterators we can pull from
        labeled_iter = iter(self.labeled_indices)
        unlabeled_iter = iter(self.unlabeled_indices)

        for _ in range(self.num_batches_per_epoch):
            batch_indices = []
            
            try:
                # Grab the labeled samples for this batch
                for _ in range(self.num_labeled_per_batch):
                    batch_indices.append(next(labeled_iter))

                # Grab the unlabeled samples for this batch
                for _ in range(self.num_unlabeled_per_batch):
                    batch_indices.append(next(unlabeled_iter))

            except StopIteration:
                # Ran out of data before finishing all planned batches
                print("Warning: Exhausted data before completing epoch. Consider adjusting batch size or iteration count.")
                break

            # Mix up the order so labeled/unlabeled samples are interleaved
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        """Number of batches this sampler will produce per epoch."""
        return self.num_batches_per_epoch