import matplotlib.pyplot as plt
import seaborn as sns


def draw_heatmap(data, title):
    sns.set(style='whitegrid')

    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30, color='blue')

    plt.title('Data Distribution' + title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.savefig('distribution/' + title + '.png')
    plt.close()


