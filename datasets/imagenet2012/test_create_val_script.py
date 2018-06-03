lines = open("val_script.sh").readlines()

first_set = set()

for line in lines:
    if line.startswith("mkdir"):
        first_set.add(line.replace("mkdir -p images/val/","").rstrip("\n"))

# print(first_set)

lines = open("val_truth.sh").readlines()

second_set = set()

for line in lines:
    if line.startswith("mkdir"):
        second_set.add(line.replace("mkdir -p ","").rstrip("\t\n"))

# print(second_set)

# first_set.add("laranja")
# second_set.add("banana")

print("first set less second set:", first_set - second_set)
print("second set less first set:", second_set - first_set)
