
import random
from datasets import load_dataset

# AG News labels
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Expert is strong in these classes
STRONG_CLASSES = {1, 2} 


def simulated_expert(true_label):
    if true_label in STRONG_CLASSES:
        # expert is usually right
        if random.random() < 0.9:
            return true_label
    else:
        # expert is basically guessing
        if random.random() < 0.3:
            return true_label

    # when expert gets it wrong 
    other_labels = [l for l in LABEL_NAMES if l != true_label]
    return random.choice(other_labels)



# Load dataset (real AG News test set)

dataset = load_dataset("fancyzhx/ag_news")
test_labels = dataset["test"]["label"]


# Run the expert on every test example

correct = 0
per_class_correct = {label: 0 for label in LABEL_NAMES}
per_class_total = {label: 0 for label in LABEL_NAMES}

for true_label in test_labels:
    expert_guess = simulated_expert(true_label)

    per_class_total[true_label] += 1
    if expert_guess == true_label:
        correct += 1
        per_class_correct[true_label] += 1


# Report results

overall_accuracy = correct / len(test_labels)
print(f"Overall expert accuracy: {overall_accuracy:.2f}")

print("\nPer-class accuracy:")
for label, name in LABEL_NAMES.items():
    acc = per_class_correct[label] / per_class_total[label]
    print(f"{name}: {acc:.2f}")
