import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, mean_scores_10, reward):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.ylim(ymin=0)
    plt.subplot(2, 1, 1)

    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(mean_scores_10)

    plt.subplot(2, 1, 2)
    plt.ylabel('Reward')
    plt.plot(reward)
    plt.autoscale(True)

    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)





