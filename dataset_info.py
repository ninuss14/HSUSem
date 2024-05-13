import csv
from collections import defaultdict
import matplotlib.pyplot as plt


def count_images_per_class(metadata_file):
    class_id_counts = defaultdict(int)

    with open(metadata_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_id = int(row['ClassId'])
            class_id_counts[class_id] += 1

    return class_id_counts


def read_image_metadata(metadata_file):
    widths = []
    heights = []
    resolutions = []
    ratios = []

    with open(metadata_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            width = int(row['Width'])
            height = int(row['Height'])
            resolution = width * height
            ratio = width / height

            widths.append(width)
            heights.append(height)
            resolutions.append(resolution)
            ratios.append(ratio)

    return widths, heights, resolutions, ratios


def plot_histogram(data, title, xlabel, ylabel, range):
    plt.hist(data, bins=50, range=range, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def main(file_name):
    class_id_counts = count_images_per_class(file_name)
    sorted_class_id_counts = sorted(class_id_counts.items(), key=lambda x: x[0])
    for class_id, count in sorted_class_id_counts:
        print(f"Class ID {class_id}: {count} pictures")
    # Plot histograms
    widths, heights, resolutions, ratios = read_image_metadata(file_name)
    plot_histogram(widths, 'Width Histogram ' + file_name, 'Width', 'Frequency', (30, 50))
    plot_histogram(heights, 'Height Histogram ' + file_name, 'Height', 'Frequency', (30, 50))
    plot_histogram(resolutions, 'Resolution Histogram ' + file_name, 'Resolution', 'Frequency', (0, 15000))
    plot_histogram(ratios, 'Width-to-Height Ratio Histogram ' + file_name, 'Width-to-Height Ratio', 'Frequency', (0.5, 1.3))

    # Extract class IDs and frequencies from the dictionary
    class_ids = list(class_id_counts.keys())
    frequencies = list(class_id_counts.values())
    plt.bar(class_ids, frequencies, color='blue', alpha=0.7)
    plt.title("Number of Images per Class")
    plt.xlabel("Class ID")
    plt.ylabel("Frequency")
    plt.xticks(range(0, 43))  # Set x-axis ticks from 1 to 43
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability if needed
    plt.show()

if __name__ == "__main__":
    print("TRAIN :")
    main('dataset/Train.csv')
    print("TEST :")
    main('dataset/Test.csv')
