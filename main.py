"""
Created on Fri Nov 27 16:00:47 2020

@author: hien
"""
import argparse
from train import train
from test_game import _test
from interactive import interactive

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Procon""")
    parser.add_argument("--file_name", default = "input.txt")
    parser.add_argument("--type", default = "1")
    parser.add_argument("--run", type=str, default="train")   
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of state per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr_actor", type=float, default=1e-5)
    parser.add_argument("--lr_critic", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--discount", type=float, default=1)   
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--replay_memory_size", type=int, default=20000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--n_games", type=str, default=1)
    parser.add_argument("--n_maps", type=str, default=1900)
    parser.add_argument("--n_epochs", type=str, default=1000000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--show_screen", type=str, default=False)
    parser.add_argument("--load_checkpoint", type=str, default=False)
    parser.add_argument("--saved_checkpoint", type=str, default=True)   
    
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    opt = get_args()
    # print(opt.)
    if opt.run == "test":
        _test(opt)
    if opt.run == "train":
        train(opt)
    if opt.run == "interactive":
        interactive(opt)
    
