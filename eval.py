from support import Timer
from support import Variable, torch


def create_eval_batch(data, char2id, start_i, seq_length=200, batch_size=1):
    # For each batch, sample a slice of the data string and convert to char ids
    batch = []
    for batch_id in range(batch_size):
        i_start_seq = start_i + (seq_length * batch_id)
        substring = data[i_start_seq:i_start_seq+seq_length]
        batch.append([char2id[char] for char in substring])
    
    # Convert it to a torch Variable
    batch = Variable(torch.LongTensor(batch))
    
    # The input and outputs are the same, just shifted by one value
    X = batch[:, :-1]
    
    # Not entirely sure if it is necessary to have the underlying data
    # being a different object, but doing it to be safe.
    Y = batch[:, 1:].clone()
    
    return X, Y

