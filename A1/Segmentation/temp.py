import torch

# Before clearing cache, check memory
allocated_before = torch.cuda.memory_allocated()
reserved_before = torch.cuda.memory_reserved()

print(f"Allocated memory before clearing: {allocated_before / 1024**2:.2f} MB")
print(f"Reserved memory before clearing: {reserved_before / 1024**2:.2f} MB")

# Clear the cache
torch.cuda.empty_cache()

# After clearing cache, check memory again
allocated_after = torch.cuda.memory_allocated()
reserved_after = torch.cuda.memory_reserved()

print(f"Allocated memory after clearing: {allocated_after / 1024**2:.2f} MB")
print(f"Reserved memory after clearing: {reserved_after / 1024**2:.2f} MB")

# Calculate the difference
allocated_freed = (allocated_before - allocated_after) / 1024**2
reserved_freed = (reserved_before - reserved_after) / 1024**2

print(f"Memory freed (allocated): {allocated_freed:.2f} MB")
print(f"Memory freed (reserved): {reserved_freed:.2f} MB")


print(torch.cuda.memory_summary())
