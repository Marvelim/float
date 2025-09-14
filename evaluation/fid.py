import cleanfid

def calculate_fid(real_dir, fake_dir):
    

if __name__ == "__main__":
    real_dir = "/home/mli374/float/evaluation/real"
    fake_dir = "/home/mli374/float/evaluation/fake"
    fid = calculate_fid(real_dir, fake_dir)
    print(f"FID: {fid}")