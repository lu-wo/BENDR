import wandb
import yaml, json 
from model_torch import create_bendr, _make_mask, _make_span_from_seeds
from config import params
import time
import os, sys
from datasets import MultiTaskDataset 
import logging
import torch 
import fnmatch
import numpy as np 


def load_data_paths(root_dir, file_pattern):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, file_pattern):
            paths.append(os.path.join(dirpath, filename))
    return paths


def main():

    # Create model directory and Logger
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f'reports/logs/{run_id}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # change std output to file in log_dir 
    sys.stdout = open(f"{log_dir}/stdout.log", "w")
    
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")

    # Store params as json 
    with open(f"{log_dir}/params.json", 'w') as fp:
        json.dump(params, fp)

    root_dir = params['root_dir']
    window_len = params['window_len']
    buffer_size = 100 
    batch_size = params['batch_size']

    # Read the paths of participants and split 
    dirs = [os.path.join(root_dir, o) for o in os.listdir(root_dir) 
                    if os.path.isdir(os.path.join(root_dir,o))]
    dirs = [os.path.join(o, p) for o in dirs for p in os.listdir(o) 
                if os.path.isdir(os.path.join(o,p))]    
    logging.info(dirs)
    nb_dirs = len(dirs)
    np.random.seed(42)
    nums = np.random.permutation(nb_dirs)
    train_dirs = [dirs[i] for i in nums[:int(0.8*nb_dirs)]]
    val_dirs = [dirs[i] for i in nums[int(0.8*nb_dirs):int(0.9*nb_dirs)]]
    test_dirs = [dirs[i] for i in nums[int(0.9*nb_dirs):]]
    file_pattern="*stream*.csv"
    train_paths = []
    for d in train_dirs:
        train_paths += load_data_paths(d, file_pattern)
    val_paths = []
    for d in val_dirs:
        val_paths += load_data_paths(d, file_pattern)
    test_paths = []
    for d in test_dirs:
        test_paths += load_data_paths(d, file_pattern)
    logging.info(f"Found {len(train_paths)} training files.")
    logging.info(f"Found {len(val_paths)} validation files.")
    logging.info(f"Found {len(test_paths)} test files.")

    train_dataset = MultiTaskDataset(paths=train_paths, window_len=window_len, buffer_size=buffer_size)
    val_dataset = MultiTaskDataset(paths=val_paths, window_len=window_len, buffer_size=buffer_size)
    test_dataset = MultiTaskDataset(paths=test_paths, window_len=window_len, buffer_size=buffer_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, workers=0)

    model = create_bendr(params)
    logging.info("Created model.")
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_weight_decay'])

    best_val = 0    
    for epoch in range(params['epochs']):
        logging.info(f"Epoch {epoch}")

        # Training loop
        train_loss_acc = 0
        for batch_idx, batch in enumerate(train_loader):
            x = batch

            x = x.permute(0, 2, 1)
            z = model.encoder(x)

            if model.permuted_encodings:
                z = z.permute([1, 2, 0])
            unmasked_z = z.clone()
            batch_size, feat, samples = z.shape
            mask = _make_mask((batch_size, samples), model.mask_rate, samples, model.mask_span) 

            c = model.context_fn(z, mask)

            # logging.info(f"encoder output {c} \ncnn output {z}")

            # Select negative candidates and generate labels for which are correct labels
            negatives, negative_inds = model._generate_negatives(z)

            # Prediction -> batch_size x predict_length x predict_length
            logits = model._calculate_similarity(unmasked_z, c, negatives)
            loss = model.calculate_loss(logits, z)



            train_loss_acc += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(f"Training loss {loss.item()}")

            if batch_idx % 100 == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
        train_loss = train_loss_acc / len(train_loader)
        logging.info(f"Epoch {epoch}, Train Loss {train_loss}")

        # validation loop 
        val_loss_acc = 0
        for batch_idx, batch in enumerate(val_loader):
            x = batch
            x = x.permute(0, 2, 1)
            z = model.encoder(x)

            if model.permuted_encodings:
                z = z.permute([1, 2, 0])

            unmasked_z = z.clone()
            batch_size, feat, samples = z.shape

            mask = torch.zeros((batch_size, samples), requires_grad=False, dtype=torch.bool)
            half_avg_num_seeds = max(1, int(samples * model.mask_rate * 0.5))
            if samples <= model.mask_span * half_avg_num_seeds:
                raise ValueError("Masking the entire span, pointless.")
            mask[:, _make_span_from_seeds((samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int),
                                                model.mask_span)] = True

            # logging.info(f'x : {x.size()}')
            # logging.info(f'x : {z.size()}')
            # logging.info(f'mask : {mask.size()}')
            
            c = model.context_fn(z, mask)

            # Select negative candidates and generate labels for which are correct labels
            negatives, negative_inds = model._generate_negatives(z)

            logits = model._calculate_similarity(unmasked_z, c, negatives)
            loss = model.calculate_loss(logits, z)
            val_loss_acc += loss.item()

            logging.info(f"Validation loss {loss.item()}")

            if batch_idx % 100 == 0:
                logging.info(f"Validation Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
        val_loss = val_loss_acc / len(val_loader)
        logging.info(f"Validation Epoch {epoch}, Loss {val_loss}")

        if val_loss < best_val: 
            best_val = val_loss
            torch.save(model.state_dict(), f"{log_dir}/best_model.pt")

    # test loop 
    test_loss_acc = 0
    for batch_idx, batch in enumerate(test_loader):
        x = batch
        x = x.permute(0, 2, 1)
        z = model.encoder(x)

        if model.permuted_encodings:
            z = z.permute([1, 2, 0])

        unmasked_z = z.clone()
        batch_size, feat, samples = z.shape

        mask = torch.zeros((batch_size, samples), requires_grad=False, dtype=torch.bool)
        half_avg_num_seeds = max(1, int(samples * model.mask_rate * 0.5))
        if samples <= model.mask_span * half_avg_num_seeds:
            raise ValueError("Masking the entire span, pointless.")
        mask[:, _make_span_from_seeds((samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int),
                                            model.mask_span)] = True

        # logging.info(f'x : {x.size()}')
        # logging.info(f'x : {z.size()}')
        # logging.info(f'mask : {mask.size()}')
        
        c = model.context_fn(z, mask)

        # Select negative candidates and generate labels for which are correct labels
        negatives, negative_inds = model._generate_negatives(z)

        logits = model._calculate_similarity(unmasked_z, c, negatives)
        loss = model.calculate_loss(logits, z)
        test_loss_acc += loss.item()

        logging.info(f"Test loss {loss.item()}")

        if batch_idx % 100 == 0:
            logging.info(f"Test Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

    test_loss = test_loss_acc / len(test_loader)
    logging.info(f"Test Loss {test_loss}")

    logging.info("--Finished logging.")
    
if __name__ == "__main__":
    main()
