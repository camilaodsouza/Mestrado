import scipy.io
meta_mat = scipy.io.loadmat("ILSVRC2012_devkit_t12/data/meta.mat")

ids = [m[0][0][0][0] for m in meta_mat['synsets']]
wnids = {m[0][0][0][0]: m[0][1][0] for m in meta_mat['synsets']}
synsets = {m[0][0][0][0]: m[0][2][0] for m in meta_mat['synsets']}

f = open("val_script.sh", "w+")

for id in ids:
    if id <= 1000:
        print("{0},{1},{2}".format(id, wnids[id], synsets[id]))
        f.write("mkdir -p images/val/" + wnids[id] + "\n")

labels = open("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt").readlines()

# print(sorted(labels)[::-1])

for i in range(len(labels)):
    f.write("mv images/val/ILSVRC2012_val_{0:08d}.JPEG images/val/".format(i+1) + wnids[int(labels[i])] + "\n")
