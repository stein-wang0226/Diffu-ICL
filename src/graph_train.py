import os
import uuid
import networkx as nx
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import random
from torch_geometric.utils import to_networkx
from torch.nn.utils.rnn import pad_sequence

from graph_tasks import get_graph_task_sampler
from graph_samplers import get_graph_data_sampler
from schema import schema
from graph_models import build_graph_model



import wandb

torch.backends.cudnn.benchmark = True

def tokenize_graph(graph):
    """Converts a graph to a token sequence using an Eulerian path."""
    G = to_networkx(graph, to_undirected="upper").to_undirected()
    
    # 1. Create a local vocabulary for the nodes in this graph to prevent data leakage.
    # Shuffling ensures that the raw node ID has no correlation with its degree.
    nodes = list(G.nodes())
    random.shuffle(nodes)
    # We reserve token 0 for padding, so local tokens start from 1.
    node_to_token = {node: i + 1 for i, node in enumerate(nodes)}

    # 2. Find a starting node and create the Eulerian path.
    start_node = 0
    if len(nodes) > 0:
        start_node = random.choice(nodes)

    if not nx.is_eulerian(G):
        G = nx.eulerize(G)

    path = list(nx.eulerian_path(G, source=start_node))
    
    if not path:
        # Handle single-node graphs
        if not nodes:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor([node_to_token[start_node]], dtype=torch.long)

    # 3. Create the token sequence using the local vocabulary.
    tokens = [node_to_token[node] for edge in path for node in edge]
    return torch.tensor(tokens, dtype=torch.long)

def train(model, args):
    # Define special tokens based on n_dims from the model config
    # Node tokens are in the range [1, n_dims]
    n_dims = args.model.n_dims
    QUESTION_TOKEN_ID = n_dims + 1
    MASK_TOKEN_ID = n_dims + 2
    ANSWER_TOKEN_OFFSET = n_dims + 3

    # Assert that the vocab size in the config is set correctly to accommodate all tokens.
    # Vocab size = 1 (padding) + n_dims (nodes) + 2 (special) + n_dims (possible answers)
    expected_vocab_size = 1 + n_dims + 2 + n_dims
    assert args.model.vocab_size >= expected_vocab_size, (
        f"Model vocab_size ({args.model.vocab_size}) is not large enough. "
        f"Expected at least {expected_vocab_size} for this task."
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    data_sampler = get_graph_data_sampler(
        args.training.data,
        num_nodes=n_dims,
    )
    task_sampler = get_graph_task_sampler(
        args.training.task,
        args.training.batch_size,
        **args.training.task_kwargs,
    )
    task = task_sampler()
    # The task is now masked token prediction, so we use CrossEntropyLoss.
    loss_func = torch.nn.CrossEntropyLoss()

    pbar = tqdm(range(args.training.train_steps))

    for i in pbar:
        optimizer.zero_grad()

        # 1. Sample graphs and get their corresponding max degree targets
        graphs = data_sampler.sample_graph(batch_size=args.training.batch_size)
        targets = task.evaluate(graphs)  # Float tensor of max degrees

        # 2. Construct input_ids and labels for the masked prediction task
        input_ids_list = []
        labels_list = []

        for graph, target in zip(graphs, targets):
            graph_tokens = tokenize_graph(graph)

            # Sequence format: [graph_tokens, QUESTION_TOKEN, MASK_TOKEN]
            input_seq = torch.cat([
                graph_tokens,
                torch.tensor([QUESTION_TOKEN_ID, MASK_TOKEN_ID], dtype=torch.long)
            ])

            # Labels: -100 for all positions except the MASK position
            labels_seq = torch.full_like(input_seq, -100)
            answer_token = ANSWER_TOKEN_OFFSET + int(target.item())
            labels_seq[-1] = answer_token

            input_ids_list.append(input_seq)
            labels_list.append(labels_seq)

        # 3. Pad the batch
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids_padded != 0)

        # 4. Forward pass and loss calculation
        logits = model(input_ids_padded.cuda(), attention_mask=attention_mask.cuda(), task_type=args.training.task_type)
        
        # Reshape for CrossEntropyLoss, which expects (N, C) and (N)
        loss = loss_func(logits.view(-1, logits.size(-1)), labels_padded.cuda().view(-1))
        
        loss.backward()
        optimizer.step()

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            metrics_to_log = {"train/loss": loss.item()}

            # Correctly evaluate accuracy at the MASKED positions
            with torch.no_grad():
                # Find where the MASK_TOKEN_ID is in the input
                mask_token_positions = (input_ids_padded == MASK_TOKEN_ID).nonzero(as_tuple=True)
                
                # Get the logits and labels at these specific positions
                mask_logits = logits[mask_token_positions]
                mask_labels = labels_padded[mask_token_positions]

                predicted_tokens = torch.argmax(mask_logits, dim=-1)
                
                correct_predictions = (predicted_tokens.cpu() == mask_labels.cpu())
                accuracy = correct_predictions.sum().item() / len(correct_predictions) if len(correct_predictions) > 0 else 0.0
                
                metrics_to_log["eval/accuracy"] = accuracy
            
            # Log the entire dictionary in one call
            wandb.log(metrics_to_log, step=i)

        pbar.set_description(f"loss {loss.item():.4f}")


def main(args):
    if not args.test_run:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    # When using dream, we must use the reframed denoising task.
    # The other models still expect the old regression setup.
    if args.model.family == "dream":
        assert args.training.task_type == "diffusion_autoregressive", \
            "Dream model must use 'diffusion_autoregressive' task_type for the denoising setup."
        # The model now outputs logits for a classification task.
        model = build_graph_model(args.model)
    else:
        # This branch is now for non-dream models and requires the old regression setup.
        # For simplicity, we are focusing on the dream model implementation.
        # You would need to add back the old train_step and adapt the train loop if you want to run them.
        raise NotImplementedError("This script is now configured for Dream model's denoising task.")

    model.cuda()
    model.train()

    train(model, args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    # Note: Ensure your config file (e.g., toy_dream.yaml) has an updated vocab_size.
    # For n_dims=50, vocab_size should be at least 1+50+2+50 = 103.
    # example: model.vocab_size=128

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
