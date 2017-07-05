# TODO: There is a lot of code overlap between the functions here and the
#       training ones. Try to create separate functions that will be shared
#       by both.

from support import Timer
from support import Variable, torch


# ==============================================================================
#                                                              CREATE_EVAL_BATCH
# ==============================================================================
def create_eval_batch(data, char2id, start_i, seq_length=200, batch_size=1):
    """ Returns a two torch Variables X, Y, each of size
            batch_size x (length-1)

        Where each row is a substring from the `data` string
        represented as a sequence of character ids.

    """
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


# ==============================================================================
#                                                                  EVAL_SEQUENCE
# ==============================================================================
def eval_sequence(model, X, Y):
    """ Given a model, the input and target labels representing a batch of
        sequences of characters, it returns the average loss of that sequence.

    Args:
        model:      (Model object)
                    The model containing the neural net architecture.
        X:          (torch Variable)
                    The input tensor of shape: [batch, sequence_length]
        Y:          (torch Variable)
                    The output labels tensor of shape: [batch, sequence_length]

    Returns:
        Returns the average loss over the batch of sequences.
    """
    # Get dimensions of input and target labels
    msg = "X and Y should be shape [n_batch, sequence_length]"
    assert len(X.size()) == len(Y.size()) == 2, msg
    batch_size, sample_length = X.size()

    # Store the original training mode, and turn training mode Off
    train_mode = model.training
    model.train(False)
    
    # Initialize hidden state and reset the accumulated gradients
    hidden = model.init_hidden(batch_size)
    model.zero_grad() # TODO: This might be irrelevant for evaluation
    
    # Loop through each item in the sequence
    loss = 0
    for i in range(sample_length):
        output, hidden = model(X[:, i], hidden)
        loss += model.loss_func(output, Y[:, i])
    
    # Restore the original training mode
    model.train(train_mode)

    # Return the average loss over the batch of sequences
    return loss.data[0] / sample_length


# ==============================================================================
#                                                                     EVAL_MODEL
# ==============================================================================
def eval_model(model, data, char2id, seq_length=200, batch_size=32):
    """ Evaluates the model on the desired evaluation data.
        Returns a tuple:
            avg_loss, total_time
    """
    data_len = len(data)
    n_steps = int(data_len / seq_length/ batch_size)
    
    # batch_size = 256
    
    timer = Timer()
    total_loss = 0
    
    for step in range(n_steps):
        # Create an evaluation batch of sequences
        i_start = step*seq_length*batch_size
        X, Y = create_eval_batch(data, char2id, start_i=i_start,
                                 seq_length=seq_length,
                                 batch_size=batch_size)

        # Perform an evaluation step on batch of sequences
        loss = eval_sequence(model, X, Y)
        total_loss += loss
        
    # Return the average loss, and total time
    avg_loss = total_loss / n_steps
    return avg_loss, timer.elapsed()

