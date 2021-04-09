from networks.resolution.vdsr import vdsr

if __name__ == "__main__":
    model = vdsr((250, 250, 3), 64)
    model.load_weights("./weights/vdsr_epoch50.hdf5")
