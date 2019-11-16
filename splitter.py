#!/usr/bin/env python3

remaining_data_len = 21

n_batches = 10

batch_sizes = [-1 for _ in range(n_batches)]
for i, b in enumerate(range(n_batches, 0, -1)):
    batch_sizes[i] = round(remaining_data_len / b)
    remaining_data_len -= batch_sizes[i]

print(batch_sizes)
print(sum(batch_sizes))
