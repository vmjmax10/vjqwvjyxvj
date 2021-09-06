# Standrad lib imports
import argparse
import time





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
    parser.add_argument(
        "--face_db_path", 
        default="face_db", 
        type=str, 
        help="Load pre face images of the live session persons"
    )
    return parser


class FaceRecognition(object):

    def __init__(self, face_db_path, face_db_blobs=None) -> None:
        super().__init__()

        self.load_face_recognition_model()
        self.compute_face_db_embeddings()

    def load_face_recognition_model():
        pass


        










def main(args):

    """
        args: Either a dictionary or argument parsed args
    """
    if not isinstance(args, dict):
        args = vars(args)
    
    assert isinstance(args, dict), "Argument must be of type dict"
    









if __name__ == "__main__":
    args = make_parser().parse_args()
    print("Live session begin at time => ", time.time())
    main(args)
    print("Live session end at time => ", time.time())