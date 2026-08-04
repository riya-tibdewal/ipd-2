"""Offline Script to Register Face Embeddings"""
import os
import pickle

def register_faces(dataset_dir: str = "data/students", output_pkl: str = "face_database.pkl"):
    database = {}
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    # Process student sub-folders and store embeddings
    print(f"Scanning directory: {dataset_dir}")
    for student_name in os.listdir(dataset_dir):
        student_path = os.path.join(dataset_dir, student_name)
        if os.path.isdir(student_path):
            database[student_name] = [] # List of 512-d embeddings
            
    with open(output_pkl, "wb") as f:
        pickle.dump(database, f)
    print(f"Face database saved to {output_pkl} with {len(database)} entries.")

if __name__ == "__main__":
    register_faces()