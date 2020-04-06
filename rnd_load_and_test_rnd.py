import torch
import matplotlib.pyplot as plt
from module.rnd import RandomNetworkDistillation
from module.load_and_split_mnist_dataset import concat_data


if __name__ == "__main__":
    # subplot_id = 1
    # for data_id in range(0, 8, 2):
    #     labels = []
    #     intrinsic_rewards = []

    #     td, vd = concat_data(
    #         list(range(10))[data_id:data_id+4], mode="tensor")

    #     for model_id in range(0, 8, 2):
    #         rnd = RandomNetworkDistillation()
    #         rnd.load(
    #             f"model/subenv_{model_id}-{model_id+4}/rnd_model", load_checkpoint=True)

    #         print(rnd.get_intrinsic_reward(vd))
    #         labels.append(f"{model_id}-{model_id+3}")
    #         intrinsic_rewards.append(rnd.get_intrinsic_reward(vd))
    #         del rnd

    #     del td, vd

    #     plt.subplot(2, 2, subplot_id)
    #     plt.title(f"data label {data_id}-{data_id+3}")
    #     plt.ylim((-1, 1))
    #     plt.plot(labels, intrinsic_rewards, "gs--")
    #     plt.xlabel("model label")
    #     plt.ylabel("intrinsic reward")
    #     subplot_id += 1


    # plt.suptitle("Intrinsic reward in dynamic dataset")
    # plt.show()

# ==============

    # subplot_id = 1
    # for data_id in range(10):
    #     labels = []
    #     intrinsic_rewards = []
    #     td, vd = concat_data([data_id], mode="tensor")
    #     for model_id in range(10):
    #         rnd = RandomNetworkDistillation()
    #         rnd.load(
    #             f"model/subenv_{model_id}/rnd", load_checkpoint=False)

    #         print(rnd.get_intrinsic_reward(vd))
    #         labels.append(f"{model_id}")
    #         intrinsic_rewards.append(rnd.get_intrinsic_reward(vd))
    #         del rnd
    #     del td, vd

    #     plt.subplot(5, 2, subplot_id)
    #     plt.title(f"data lebel {data_id}")
    #     plt.ylim((-3, 3))
    #     plt.plot(labels, intrinsic_rewards, "gs--")
    #     # plt.xlabel("data label")
    #     # plt.ylabel("intrinsic reward")
    #     subplot_id += 1

    # plt.suptitle("Intrinsic reward in dynamic dataset")
    # plt.show()

# ===

    labels = []
    intrinsic_rewards = []
    for data_id in range(0, 8, 2):
        td, vd = concat_data(
            list(range(10))[data_id:data_id+4], mode="tensor")

        rnd = RandomNetworkDistillation()
        rnd.load(
            f"model/allenv/rnd_model", load_checkpoint=True)
        print(rnd.get_intrinsic_reward(vd))
        labels.append(f"{data_id}-{data_id+3}")
        intrinsic_rewards.append(rnd.get_intrinsic_reward(vd))

    plt.suptitle("Intrinsic reward in dynamic dataset (full MNIST data model)")
    plt.plot(labels, intrinsic_rewards, "gs--")
    plt.ylim((-5, 5))
    plt.xlabel("data label")
    plt.ylabel("intrinsic reward")
    plt.show()
