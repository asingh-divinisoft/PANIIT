from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

class BuildDataset:

    def __init__(self, path, save_path):
        self.path = path
        self.save_path = save_path
        self.filenames = sorted(os.listdir(train_dir),key=lambda x: int(x.split('.')[0]))
        self.csv_path = '/home/aaditya/PycharmProjects/PANIIT/data/unzipped/training/'
        self.csv = pd.read_csv(self.csv_path + 'solution.csv')

    def open_img(self, idx):
        return Image.open(os.path.join(self.path, self.filenames[idx]))

    def get_bw(self, idx):
        img = self.open_img(idx)
        return ImageOps.grayscale(img)

    def show_img(self, idx=None, img=None):
        if idx is not None:
            plt.title(self.filenames[idx])
            if img is None:
                img = self.open_img(idx)

        if img is not None:
            plt.matshow(img, cmap='gray')
            # print(np.array(img).shape)

    def augmentation(self):
        total = len(self.filenames)
        idxs = list(range(total))
        cur_total = 1
        new_csv_data = []

        for i in tqdm(idxs[:4000]):
            # 4 images for 1 image
            category = self.csv.category[i]
            bw_img = self.get_bw(i)
            imgs = [
                bw_img,
                bw_img.transpose(Image.FLIP_LEFT_RIGHT)
            ]

            top_down_imgs = []
            for img in imgs:
                top_down_imgs.append(img.transpose(Image.FLIP_TOP_BOTTOM))

            imgs.extend(top_down_imgs)

            for j, img in enumerate(imgs):
                if not os.path.exists(os.path.join(self.save_path, 'train', f'{category}')):
                    os.mkdir(os.path.join(self.save_path, 'train', f'{category}'))

                img.save(os.path.join(self.save_path, 'train', f'{category}/{cur_total+j}.png'))
                new_csv_data.append({'id': cur_total+j, 'category': category})

            cur_total += 4

        pd.DataFrame(new_csv_data).to_csv(self.csv_path + 'new_train_solution.csv', index=False)

        cur_total = 1
        new_csv_data = []

        for i in tqdm(idxs[4000:]):
            category = self.csv.category[i]
            bw_img = self.get_bw(i)

            if not os.path.exists(os.path.join(self.save_path, 'test', f'{category}')):
                os.mkdir(os.path.join(self.save_path, 'test', f'{category}'))

            bw_img.save(os.path.join(self.save_path, 'test', f'{category}/{cur_total + j}.png'))
            new_csv_data.append({'id': cur_total + j, 'category': category})

            cur_total += 4

        pd.DataFrame(new_csv_data).to_csv(self.csv_path + 'new_test_solution.csv', index=False)

    # def reorganize(self):


if __name__ == '__main__':
    PATH = '/home/aaditya/PycharmProjects/PANIIT/data'
    train_dir = os.path.join(PATH, 'unzipped/training/training_data')
    custom_dir = os.path.join(PATH, 'custom_bw')
    data = BuildDataset(train_dir, custom_dir)
    id = 301
    # data.show_img(idx=id, img=data.get_bw(id).transpose(Image.FLIP_TOP_BOTTOM))
    # print(data.csv.category[id])
    # plt.show()
    data.augmentation()
