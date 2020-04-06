import torch
import matplotlib.pyplot as plt
from module.gan import GenerativeAdversarialNetwork
from module.load_and_split_mnist_dataset import concat_data


if __name__ == "__main__":
    # subplot_id = 1
    # for data_id in range(0, 8, 2):
    #     labels = []
    #     discriminator_output = []

    #     td, vd = concat_data(
    #         list(range(10))[data_id:data_id+4], mode="tensor")

    #     for model_id in range(0, 8, 2):
    #         gan = GenerativeAdversarialNetwork(
    #             save_path=f"model/subenv_{model_id}-{model_id+4}/gan")
    #         gan.load()

    #         print(gan.get_discriminator_output(vd))
    #         labels.append(f"{model_id}-{model_id+3}")
    #         discriminator_output.append(gan.get_discriminator_output(vd))
    #         del gan

    #     del td, vd

    #     plt.subplot(2, 2, subplot_id)
    #     plt.title(f"data label {data_id}-{data_id+3}")
    #     plt.ylim((-0.1, 1.1))
    #     plt.plot(labels, discriminator_output, "gs--")
    #     for a, b in zip(labels, discriminator_output):
    #         plt.text(a, b, str(f"{b.item():.2f}"))
    #     plt.xlabel("model label")
    #     plt.ylabel("discriminator output")
    #     subplot_id += 1

    # plt.suptitle("Discriminator output in dynamic dataset")
    # plt.show()

# ===

    subplot_id = 1
    for model_id in range(0, 8, 2):
        gan = GenerativeAdversarialNetwork(
            save_path=f"model/subenv_{model_id}-{model_id+4}/gan")
        gan.load()

        labels = []
        discriminator_output = []
        for data_id in range(0, 8, 2):

            td, vd = concat_data(
                list(range(10))[data_id:data_id+4], mode="tensor")
            print(gan.get_discriminator_output(vd))
            labels.append(f"{data_id}-{data_id+3}")
            discriminator_output.append(gan.get_discriminator_output(vd))
        del td, vd
        plt.subplot(2, 2, subplot_id)
        plt.title(f"model label {model_id}-{model_id+3}")
        plt.ylim((-0.1, 1.1))
        plt.plot(labels, discriminator_output, "rs--")
        for a, b in zip(labels, discriminator_output):
            plt.text(a, b, str(f"{b.item():.2f}"))
        plt.xlabel("data label")
        plt.ylabel("discriminator output")
        subplot_id += 1
    del gan

    plt.suptitle("Discriminator output in dynamic dataset")
    plt.show()