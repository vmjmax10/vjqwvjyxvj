# Standrad lib imports
import argparse

def make_parser():
    parser = argparse.ArgumentParser("LIVE SESSION RUN SETTINGS")
    
    parser.add_argument(
        "--webcam", default=0, type=int, help="webcam id for the live session"
    )
    parser.add_argument(
        "--perform_liveness_test", 
        default=1, 
        type=int, 
        help="Wheter to perform liveness test or not"
    )
    parser.add_argument(
        "--verify_face_from_id", 
        default=1, 
        type=int, 
        help="Wheter to perform face verification from the ID"
    )
    parser.add_argument(
        "--liveness_after_re_init", 
        default=0, 
        type=int, 
        help="Wheter to perform liveness test after lost tracking in a session"
    )

    return parser


def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
   



    